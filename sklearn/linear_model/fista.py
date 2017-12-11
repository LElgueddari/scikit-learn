import numpy as np
from scipy.linalg import svdvals
import warnings


class Grad2D(object):
    """ Standard 2D gradient class

    This class defines the gradient operator of the following equation:

        (1 / 2) * ||y - Xw||^2_2

    Parameters
    ----------
    y : np.ndarray of shape (n_y, m_y)
        input array of the observed data
    X  : nd.array of shape (n_y, m_x)
        is a matrix that defines a dot operation
    gram: nd.array of shape(m_x, m_x)
        is a matrix that defines the operation X.T.dot(X), Accelarate
        the calculation of the gradient
    cov: nd.array of shape (m_x, m_y)
        is a matrix that defines the operation X.T.dot(y). Accelarate
        the calculation of the gradient
    """
    def __init__(self, y, X, gram=None, cov=None):
        """ Initilize the Grad2D class.
        """
        n_x, m_x = X.shape
        n_y, m_y = y.shape
        if not(n_x == n_y):
            raise ValueError('Matrix multiplication not aligned', X.T.shape,
                             y.shape)
        self.y = y
        self.X = X

        if gram is None:
            self.Gram = np.dot(np.conj(X.T), X)
        else:
            self.Gram = gram

        if cov is None:
            self.cov = np.dot(np.conj(X.T), y)
        else:
            self.cov = cov
        self.x_shape = (m_x, m_y)
        self.n_samples = n_x
        self.get_spec_rad()

    def get_grad(self, x):
        """ Get the gradient step

        This method calculates the gradient step from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray gradient value

        Notes
        -----

        Calculates M^T (MX - Y) = M^TMX - M^TY = GramX - cov
        """
        self.grad = np.dot(self.Gram, x)
        self.grad -= self.cov

    def get_spec_rad(self):
        self.spec_rad = svdvals(self.Gram)[0]
        self.inv_spec_rad = 1.0 / self.spec_rad

    def MtMX(self, x):
        return np.dot(self.Gram, x)


class SoftThreshold(object):
    """ Soft threshold proximity operator

    This class defines the threshold proximity operator

    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    """
    def __init__(self, weights):
        self.weights = weights

    def op(self, data, extra_factor=1.0):
        """ Operator

        This method returns the input data thresholded by the weights

        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        DictionaryBase thresholded data

        """
        threshold = self.weights * extra_factor
        data *= np.maximum(1 - threshold / np.maximum(np.finfo(np.float32).eps,
                           np.abs(np.copy(data))), 0)
        return data


class FISTA(object):
    """ Forward-Backward optimisation

    This class implements standard forward-backward optimisation with an the
    option to use the FISTA speed-up

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    grad : class
        Gradient operator class
    prox : class
        Proximity operator class
    cost : class
        Cost function class
    lambda_init : float
        Initial value of the relaxation parameter
    lambda_update :
        Relaxation parameter update method
    use_fista : bool
        Option to use FISTA (default is 'True')
    auto_iterate : bool
        Option to automatically begin iterations upon initialisation (default
        is 'False')
    """

    def __init__(self, grad, prox, t=1.):
        self.grad = grad
        self.prox = prox
        self.t_prev = t
        self.t = t

    def fit(self, x=None, max_iter=1500):
        """ Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is '150')
        """
        # if self.grad.spec_rad > 10**3 and max_iter < 10**4:
        #     print(self.grad.spec_rad)
        #     warnings.warn('Lipschitz constant is to high, incresing max iter')
        #     max_iter = 10**int(np.log10(self.grad.spec_rad))

        if x is None:
            z_old = np.zeros(self.grad.x_shape).astype(self.grad.y.dtype)
            x_old = np.zeros(self.grad.x_shape).astype(self.grad.y.dtype)

        for i in xrange(max_iter):
            self.grad.get_grad(z_old)
            self.grad.grad /= self.grad.spec_rad
            z_old -= self.grad.grad
            # y_old = z_old - self.grad.grad / self.grad.spec_rad

            # Step 2 from alg.10.7.
            x_new = self.prox.op(z_old, extra_factor=self.grad.inv_spec_rad)

            # Steps 3 and 4 from alg.10.7.

            self.t_prev = self.t
            self.t = (1 + np.sqrt(4 * self.t_prev ** 2 + 1)) * 0.5
            lambda_ = (self.t_prev - 1) / self.t

            # Step 5 from alg.10.7.
            z_old = x_new - lambda_ * (x_new - x_old)

            # Update old values for next iteration.
            x_old = x_new

        self.coef_ = z_old .T
