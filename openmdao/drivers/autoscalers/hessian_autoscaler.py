import numpy as np

from openmdao.drivers.autoscalers.autoscaler import Autoscaler
from openmdao.vectors.optimizer_vector import OptimizerVector
from openmdao.core.driver import Driver
from openmdao.core.constants import INF_BOUND
from openmdao.utils.om_warnings import issue_warning
        
from scipy.sparse.linalg import eigs, LinearOperator
from .arnoldi_algorithm import sort_eigs, extract_eigpairs

class HessianAutoscaler(Autoscaler):
    """Transform optimizer variables between model and optimizer spaces.

    This is the default Autoscaler in OpenMDAO that scales optimization variables based
    on matrix design-variable transformation, x = M y.
    
    x is the model variable, y is the optimizer variable, and the matrix M is the scaler/preconditioner
    
    """
    
    def __init__(self, num_modes, num_krylov = None, grad_fd_step=1e-6, use_scipy_eigs=False, start_vec = None):
        """
        Initialize the HessianAutoscaler with the specified parameters.
        
        Parameters
        ----------
        num_modes : int
            The number of eigenpairs used for the Hessian-based scaling
        num_krylov : int, optional
            The number of Krylov iterations for the Arnoldi method (default is 2*num_modes)
        grad_fd_step : float, optional
            The step size for finite difference approximation (default is 1e-6)
        use_scipy_eigs : bool, optional
            If True, use Scipy library to compute the eigenpairs (default is False)
        start_vec : ndarray, optional
            The starting vector used in the Arnoldi algorithm (default: random if start_vec = None)
        """
        
        super().__init__()
        self.m = num_modes
        self.grad_fd_step = grad_fd_step
        self.use_scipy_eigs = use_scipy_eigs
        self.b = start_vec
        if num_krylov is None:
            num_krylov = 2*num_modes
        self.max_iter = num_krylov
        self.eigenvals_m = None
        self.eigenvecs_m = None
        self.num_grad_evals = 0
        
    def setup(self, driver: 'Driver'):
        """
        Perform setup of autoscaler during final setup of the problem.

        Parameters
        ----------
        driver : Driver
            The driver associated with this autoscaler.
        model_has_run: bool
            True if setup is being called after the model has been run. If not,
            and setup requires run model, it will be executed within setup.
        """
        super().setup(driver)
        
        # # ensure that only 1 design variable is currently considered
        dv_names = [name for name, meta in driver._designvars.items()
            if not meta.get('discrete', False)]
        
        if not dv_names:
            raise RuntimeError("HessianAutoscaler requires at least one continuous design variable.")
        
        x0_vec = driver._vectors['design_var']
        x0_vec.update_from_model(driver, driver_scaling=False)
        x0 = x0_vec.asarray().copy()
        self.n_vars = x0.size
        
        if self.m > self.n_vars:
            raise RuntimeError(f"Number of modes cannot exceed the number of design variables. There are {self.n_vars} design variables and {self.m} modes.")
        
        # store the slices and sizes of each design variable for later use in apply_jac_scaling(...)
        self._dv_slices = {}
        self._dv_sizes = {}
        for name, meta in x0_vec._meta.items():
            self._dv_slices[name] = meta['slice']
            self._dv_sizes[name] = meta['size']
        
        def obj_grad(x):
            return self._gradfunc(driver, x)
        eigenvals, self.eigenvecs_m = self._get_eigen_decomp(gradfunc=obj_grad,
                                                                    m=self.m, x=x0, 
                                                                    use_scipy_eigs=self.use_scipy_eigs, 
                                                                    b=self.b)
        eigenvals = self._clean_eigenvals(eigenvals)
        self.eigenvals_m = eigenvals.copy()
        
        # reset the model to x0 after the perturbations in _get_eigen_decomp(...)
        x0_vec.set_data(x0, driver_scaling=False)
        driver._set_design_vars(driver_scaling=False)
        driver._run_solve_nonlinear()
        
        self._has_scaling = True
                
    def _gradfunc(self, driver, x):
        """
        Extracts the gradient of the objective function at a design point
        
        Parameters
        ---------
        driver : Driver
            The driver associated with this autoscaler.
        x : ndarray
            design variables in the physical (model) space
        ndarray
            A vector of the objective gradients
        """
        dv_vec = driver._vectors['design_var']
        dv_vec.set_data(x, driver_scaling=False)
        driver._set_design_vars(driver_scaling=False)

        driver._run_solve_nonlinear()
        
        # ensure only one objective is defined
        if len(driver._objs) != 1:
            raise RuntimeError("HessianAutoscaler currently supports only one objective.")

        obj_names = list(driver._objs)
        dv_names = list(driver._designvars)

        totals = driver._compute_totals(
            of=obj_names,
            wrt=dv_names,
            return_format='array',
            driver_scaling=False,
        )
        self.num_grad_evals += 1
        return totals[0, :].copy()
    
    def _compute_Hv(self, gradfunc, x, grad_fd_step):
        """ Compute the matrix-vector (mv) product using Forward Diffrerence (FD)
        approximation. The Hessian is then formed based on mv as a linear operator.
        
        Parameters
        ----------
        gradfunc : vec
            A function that returns the gradient of the objective function in vector form.
        grad_fd_step : scalar
            the step size for FD approximation.
            
        Returns
        -------
        hessian : LinearOperator
            The objective Hessian as a Linear Operator
        """
        x = x.flatten()     # convert x to 1D vector to make it work with v
        h = grad_fd_step                                # FD perturbation
        g = gradfunc(x)
        def mv(v):
            """This subfunction performs the forward difference in the direction
            of an arbitrary vector v.
            """
            g_eps_h = gradfunc(x + h*v) # compute Hessian-vec product with FD
            
            return (g_eps_h - g)/h
        
        n = len(x)
        hessian = LinearOperator((n,n), matvec = mv)   # provide the hessian as a linear operator
    
        return hessian
    
    def _get_eigen_decomp(self, gradfunc, m, x, use_scipy_eigs, b):
        """
        Compute the eigevalue decomposition using Arnoldi method
        
        Parameters
        ----------
        gradfunc : vec
            A function that returns the gradient of the objective function in vector form.
        m : int
            The number of eigenpairs used for the Hessian-based scaling
        x : vec
            Vector of (unscaled) design variables in the model space
        use_scipy_eigs : bool
            If True, use Scipy library to compute the eigenpairs. This choice
            is more expensive and more accurate. If False, the cheaper, local impelementation 
            of the Arnoldi algorithm will be used to estimate the eigenpairs
        b : vec
            The starting vector used in the Arnoldi algorithm
            
        Returns
        --------
        eigenvals_m : arr
            Array of m largest (in magnitude) eigenvalues
        eigenvecs_m : arr
            Matrix of the eigenvectors corresponding to eigenvals_m
        """
        
        n_vars = len(x)
        
        Hv = self._compute_Hv(gradfunc, x, self.grad_fd_step)
        if b is None:
            b = np.random.rand(n_vars)
        
        if use_scipy_eigs:
            # use the Scipy library to compute the eigenpairs
            vals, vecs = eigs(Hv, k=m, which='LM', v0=b, maxiter=self.max_iter) 
            
            # extract the real parts of vals and vecs 
            real_vals = np.real(vals)               
            real_vecs = np.real(vecs)
            
            # sort the eigenvalues and eigenvectors from highest to lowest in magnitude
            eigenvals_m, eigenvecs_m = sort_eigs(unsorted_eigvals=real_vals, unsorted_eigvecs=real_vecs)
        
        else:        
            # use the local implementation of the Arnoldi algorithm
            eigenvals_m, eigenvecs_m = extract_eigpairs(A=Hv, b=b, iter=self.max_iter, m=m)
            print("eigenvalues = ", eigenvals_m)
        return eigenvals_m, eigenvecs_m
    
    def _clean_eigenvals(self, eigenvals):
        """
        Clean the eigenvalues by removing any that are nearly zero.
        
        Parameters
        ----------
        eigenvals : array
            Array of eigenvalues
            
        Returns
        -------
        cleaned_eigenvals : array
            Array of cleaned eigenvalues
        """
        if eigenvals[0] <= 1e-10:
            return eigenvals
        
        for idx, val in enumerate(eigenvals):
            if np.abs(val) <= 1e-6:
                eigenvals[idx] = 1e-6
        
        return eigenvals
    
    def _compute_matrix_M(self):
        """ 
        Computes the mapping M using m eigenvalues and eigenvectors.
        
        M = V_m * |Lambda_m|^(-1/2) * V_m^T  +  |lambda_m|^(-1/2) * (I - V_m*V_m^T)
        
        Returns
        -------
        M : ndarray
            The scaling matrix/preconditioner.
        """
        
        if np.max(np.abs(self.eigenvals_m)) <= 1e-10:
            M = np.eye(self.n_vars)
            issue_warning("HessianAutoscaler: The Hessian is nearly singular or the objective is linear. " 
                          "Using identity matrix as the scaling matrix.",
                          category=Warning)
        else:
            # if np.abs(self.eigenvals_m[self.m - 1]) <= 1e-10:
            #     raise ValueError(f"The smallest eigenvalue is approximately zero (value = {self.eigenvals_m[self.m - 1]})")
            I = np.eye(self.n_vars)
            tilde_vals = np.copy(self.eigenvals_m)
            for i in range(self.m):
                tilde_vals[i] = 1/np.sqrt(np.abs(self.eigenvals_m[i]))
                
            tilde_lambda = tilde_vals[self.m - 1]

            M = ( self.eigenvecs_m @ np.diag(tilde_vals) ) @ np.transpose(self.eigenvecs_m)
            M += tilde_lambda * ( I - self.eigenvecs_m @ np.transpose(self.eigenvecs_m) )        
        
        return M
    
    def _compute_My(self, y):
        """ 
        Compute the model design variable through the matrix-vector product My.
        
        My = V_m * ( |Lambda|^(-1/2) - |lambda_m|^(-1/2)*I_m ) * z   +   |lambda_m|^(-1/2) * y
        where z = V_m^T 
        
        Parameters
        ----------
        y : ndarray 
            The optimization variable. Note x_scaled = y
        Returns
        --------
        My : ndarray
            The unscaled design variables. That is x_model = My
        """
        
        if np.max(np.abs(self.eigenvals_m)) <= 1e-10:
            My = y
            issue_warning("HessianAutoscaler: The Hessian is nearly singular or the objective is linear. " 
                          "Using identity matrix as the scaling matrix.",
                          category=Warning)
        else:
            tilde_vals = np.copy(self.eigenvals_m)
            for i in range(self.m):
                tilde_vals[i] = 1/np.sqrt(np.abs(self.eigenvals_m[i]))
                
            tilde_lambda = tilde_vals[self.m-1]
            z = np.transpose(self.eigenvecs_m) @ y

            My = (self.eigenvecs_m @ (np.diag(tilde_vals) - tilde_lambda * np.eye(self.m))) @ z 
            My += tilde_lambda * y
        return My

    def _compute_M_inverse_x(self, x):    
        """
        Compute y = M^{-1} x without inverting any matrix

        Parameters
        ---------
        x : (1D array or vector)
            model space design variables
        m : (integer)
            number of eigenpairs used to form M
        eigenvals_m : (1D array or vector)
            m eigenvalues
        eigenvecs_m : (2D array or vector)
            eigenvectors

        Returns
        y : 1D array or vector 
            optimization variable (equivalently x_scaled)
        """
        if np.max(np.abs(self.eigenvals_m)) <= 1e-10:
            y = x
        else:
            Im = np.eye(self.m)
            tilde_vals = np.copy(self.eigenvals_m)
            for i in range(self.m):
                tilde_vals[i] = np.sqrt(np.abs(self.eigenvals_m[i]))
                
            tilde_lambda = tilde_vals[self.m-1]
            w = np.transpose(self.eigenvecs_m) @ x

            y = (self.eigenvecs_m @ (np.diag(tilde_vals) - tilde_lambda * Im)) @ w + tilde_lambda * x
            # y_0 = eigenvecs_m @ np.diag(tilde_vals) @ w
        return y


    @property
    def setup_requires_run_model(self):
        """
        Return True if this autoscaler requires that the model be in an executed state.

        The gradient of the objective function at the initial design is needed for this
        autoscaler

        This property is used to tell the driver that the run_model needs to be
        called before configuring the driver.

        Returns
        -------
        bool
            True if the driver must execute run_model before the autoscaler's configure method.
        """
        return True

    @property
    def report_after_setup(self):
        """
        Return True if the scaling report should be generated after setup() is called.

        When False (default), the scaling report is generated after the first nonlinear
        totals computation during optimization (the same timing as on master). This is
        appropriate for autoscalers whose configure() method is a no-op.

        When True, the scaling report fires immediately after configure() completes.
        Subclasses whose configure() method modifies scaling parameters should override
        this property to return True so that the report reflects the post-configure scaling.

        Returns
        -------
        bool
            True if the scaling report should be generated after configure(), False otherwise.
        """
        return True
    
    def apply_design_var_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the design variables from the model space to the optimizer space.

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='design_var'.
        """
        # Checks whether the vector is already in driver-scaled form.
        if vec.driver_scaling:
            return vec

        x_model = vec.asarray().copy()
        y = self._compute_M_inverse_x(x=x_model)
        vec.asarray()[:] = y
        vec._driver_scaling = True
        return vec


    def apply_design_var_unscaling(self, vec: 'OptimizerVector'):
        """
        Unscale the design variables from the optimizer space to the model space.

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='design_var'.
        """
        
        # check if vector is in already in model (not driver-scaled) form
        if not vec.driver_scaling:
            return vec

        y = vec.asarray().copy()
        x_model = self._compute_My(y)
        vec.asarray()[:] = x_model
        vec._driver_scaling = False
        return vec
    
    def apply_objective_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the objective from the model space to the optimizer space.

        Notes
        -----
        In the case of the Hessian Autoscaler, the scaled objective is the same as
        the unscaled objective. That is, 
            f(My) = f(x) 
                OR
            f_scaled(y) = f(x)

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='objective'.
        """
        vec._driver_scaling = True
        return vec
    
    def apply_constraint_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the constraints from the model space to the optimizer space.

        Similar to the objective, the scaled and unscaled constraints are
        the same. That is, 
            g(My) = g(x)
                OR
            g_scaled(y) = g(x)
        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='constraint'.
        """
        vec._driver_scaling = True
        return vec
    
    @staticmethod
    def update_design_vars(p):
        """
        Convert bounded design variables to linear constraints after setup but before final_setup.

        For each design variable that has finite lower or upper bounds, this function adds an
        equivalent linear constraint on that variable and removes the bounds from the design
        variable declaration.  This must be called after Problem.setup() but before run_model(), run_driver()
        or final_setup().

        Parameters
        ----------
        p : om.Problem
            The Problem instance, which must have had setup() called already.
        """

        for prom_name, meta in p.model.get_design_vars(
                recurse=True, get_sizes=False, use_prom_ivc=True).items():
            lb = meta['lower']
            ub = meta['upper']

            has_lb = np.any(np.asarray(lb) > -INF_BOUND)
            has_ub = np.any(np.asarray(ub) < INF_BOUND)

            if not has_lb and not has_ub:
                continue

            con_lower = lb if has_lb else None
            con_upper = ub if has_ub else None

            p._metadata['static_mode'] = False
            p.model.add_constraint(prom_name, lower=con_lower, upper=con_upper, linear=True,
                                ref=meta['ref'], ref0=meta['ref0'],
                                scaler=meta['scaler'], adder=meta['adder'],
                                indices=meta.get('indices'),
                                flat_indices=meta.get('flat_indices', False))
            p._metadata['static_mode'] = True
            p.model._static_responses[prom_name] = p.model._responses[prom_name]

            meta['lower'] = -INF_BOUND
            meta['upper'] = INF_BOUND
    
    def apply_mult_unscaling(self, desvar_multipliers, con_multipliers):
        """
        Unscale the Lagrange multipliers from optimizer space to model space.
        
        This method transforms active design variable bounds from the scaled optimization space back to 
        physical (model) space. It does not affect the Lagrange multipliers of active constraints.
        
        At optimality, we assume the KKT stationarity condition holds:
        
            ∇ₓf(x) + ∇ₓg(x)^T λ = 0
        
        where:
            - ∇ₓf is the gradient of the objective
            - ∇ₓg(x)^T is the Jacobian of all active constraints (each row is ∇ₓg_i^T)
            - λ is the vector of Lagrange multipliers (in optimizer-scaled)
        
        The constraint vector g(x) includes:
            - Active design variables (on their bounds, to within some tolerance)
            - Equality constraints (always active)
            - Active inequality constraints (on their bounds, to within some tolerance)
        
        Scaling Transformations
        -----------------------
        Define scaling transformations that map from unscaled (physical) space to
        scaled (optimizer) space:
        
            x_scaled = T_x(x)         (design variables)
            g_scaled = T_g(g(x))      (constraints)
            f_scaled = T_f(f(x))      (objective)
        
        Applying the chain rule to the scaled stationarity condition:
        
            ∇ₓ_scaled f_scaled + ∇ₓ_scaled g_scaled^T λ_scaled = 0
        
        The gradients in scaled space are:
        
            ∇ₓ_scaled f_scaled = (dTf/df) * ∇ₓf * (dTₓ/dx)^(-1)
            ∇ₓ_scaled g_scaled = (dTg/dg) * ∇ₓg * (dTₓ/dx)^(-1)
        
        Substituting into the scaled stationarity condition and multiplying by (dTₓ/dx)^T:
        
            (dTf/df) * ∇ₓf + (dTg/dg) * ∇ₓg^T * λ_scaled = 0
        
        Dividing by (dTf/df) and comparing with the unscaled condition ∇ₓf + ∇ₓg^T λ = 0:
        
            λ = (dTg/dg) / (dTf/df) * λ_scaled
        
        For the Hessian autoscaler, we have
        
            T_x(x) = y(x) = M^{-1} x
            T_g(g) = g(x)
            T_f(f) = f(x)
        
        The derivatives are:
        
            dT_x/dx = scaler_x = M^{-1}
            dT_g/dg = scaler_g = 1
            dT_f/df = scaler_f = 1
        
        Therefore:
        
            λ_constraint = (scaler_g / scaler_f) * λ_constraint_scaled = λ_constraint_scaled
    
            λ_bound = (scaler_x / scaler_f) * λ_bound_scaled = M^{-1} * λ_bound_scaled
        
        Parameters
        ----------
        desvar_multipliers : dict[str, np.ndarray]
            A dict of optimizer-scaled Lagrange multipliers keyed by each active design variable.
        con_multipliers : dict[str, np.ndarray]
            A dict of optimizer-scaled Lagrange multipliers keyed by each active constraint.
        
        Returns
        -------
        desvar_multipliers : dict[str, np.ndarray]
            A reference to the desvar_multipliers given on input. The values of the multipliers
            were unscaled in-place.
        con_multipliers : dict[str, np.ndarray]
            A reference to the con_multipliers given on input. The values of the multipliers
            were unscaled in-place.
        """
        # temporarily assuming design variable bounds are not present
        return desvar_multipliers, con_multipliers
    
    def apply_jac_scaling(self, jac_dict):
        """
        Scale total derivatives from model-space design variables x to optimizer-space
        variables y.

        For HessianAutoscaler:

            x = M y

        so:

            d(response)/dy = d(response)/dx @ M

        For multiple design variables, OpenMDAO may store Jacobian blocks separately
        by design variable. This method assembles the full d(response)/dx block in
        contiguous driver design-variable order, applies M, then scatters the result
        back into the original per-design-variable blocks.
        """
        if not self._has_scaling:
            return

        if not hasattr(self, '_dv_slices') or not self._dv_slices:
            return

        M = self._compute_matrix_M()

        # Use the exact slices from the OptimizerVector used to build x0 and M.
        dv_slices = self._dv_slices
        n_total = self.n_vars

        if M.shape != (n_total, n_total):
            raise RuntimeError(
                f"HessianAutoscaler M has shape {M.shape}, but the contiguous "
                f"design-variable vector has size {n_total}."
            )

        def _as_2d(block):
            arr = np.asarray(block)

            if arr.ndim == 1:
                return arr.reshape(1, -1), True

            if arr.ndim == 2:
                return arr, False

            raise RuntimeError(
                f"Cannot scale Jacobian block with {arr.ndim} dimensions."
            )

        def _scale_nested_output(jac_blocks):
            """
            Scale one nested output entry:

                jac_blocks[in_name] = d(output)/d(input)

            This modifies existing blocks in place.
            """
            
            # The first block is used to determine the number of rows in the Jacobian. 
            # scalar objective or constraint: n_row = 1, 
            # vector objective or constraint: n_row = size of the output vector
            # We assume that the columns of each block correspond to contiguous slices
            # of the design variables, as given by dv_slices. This allows us to assemble 
            # the full d(response)/dx matrix in the correct order for multiplication with M.
            first_block = None
            for in_name in dv_slices:
                if in_name in jac_blocks:
                    first_block = jac_blocks[in_name]
                    break

            if first_block is None:
                return

            first_arr, first_was_1d = _as_2d(first_block)
            n_rows = first_arr.shape[0]

            Jx = np.zeros((n_rows, n_total))

            # Assemble d(response)/dx in contiguous design-variable order.
            for in_name, slc in dv_slices.items():
                if in_name not in jac_blocks:
                    continue

                arr, was_1d = _as_2d(jac_blocks[in_name])

                if arr.shape[0] != n_rows:
                    raise RuntimeError(
                        f"Jacobian block for '{in_name}' has {arr.shape[0]} rows, "
                        f"but expected {n_rows}."
                    )

                # use slice to place the block in the correct columns of Jx
                if arr.shape[1] != slc.stop - slc.start:
                    raise RuntimeError(
                        f"Jacobian block for '{in_name}' has shape {arr.shape}, "
                        f"but expected second dimension {slc.stop - slc.start}."
                    )

                Jx[:, slc] = arr

            # Chain rule: d(response)/dy = d(response)/dx @ dx/dy = Jx @ M.
            Jy = Jx @ M

            # Scatter back into existing OpenMDAO blocks.
            for in_name, slc in dv_slices.items():
                if in_name not in jac_blocks:
                    continue

                block = jac_blocks[in_name]
                arr = np.asarray(block)
                piece = Jy[:, slc]

                if arr.ndim == 1:
                    arr[...] = piece.reshape(-1)
                else:
                    arr[...] = piece

        if not jac_dict:
            return

        first_key = next(iter(jac_dict))

        if isinstance(first_key, tuple):
            # Flat dict format:
            #     jac_dict[(out_name, in_name)] = block
            #
            # Convert to temporary nested grouping by output name.
            by_output = {}

            for (out_name, in_name), block in jac_dict.items():
                if in_name in dv_slices:
                    by_output.setdefault(out_name, {})[in_name] = block

            for blocks in by_output.values():
                _scale_nested_output(blocks)

        else:
            # Nested dict format:
            #     jac_dict[out_name][in_name] = block
            for out_name, blocks in jac_dict.items():
                _scale_nested_output(blocks)
                
        self.num_grad_evals += 1
                
        
    
