import numpy as np

def get_precode_matrix(channel_estimate,method = 'Conj'):
    """
    Args:
        channel_estimate : estimated cahnnel

    Returns:
        precode_matrix : precoding matrix
    """
    assert isinstance(channel_estimate,np.ndarray)

    if channel_estimate.ndim == 0:
        channel_estimate = np.array([channel_estimate])
    # channel_estimate = channel_estimate[np.newaxis, :]
    if method == 'ZF':
        cof1 = channel_estimate.conj().T
        # channel_estimate = np.atleast_2d(channel_estimate)

        cof2 = np.linalg.inv(channel_estimate @ channel_estimate.conj().T)

        precode_matrix = cof1 @ cof2
        return precode_matrix
    elif method == 'Conj' :
        mol = np.linalg.norm(channel_estimate)
        return channel_estimate.conj().T / mol
