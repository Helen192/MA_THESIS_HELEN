import time

import numpy as np
from scipy.stats import chi2
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class CubeMixtureModel:
    def __init__(self, means: list[np.ndarray], deviations: list[np.ndarray], weights: list[float],
                 percentile=0.80, delta=0.05, inflate_scale=5.0, inflate_full=False, sub_dims=None,border_extend_factor=2.0):  
        """
        Initializes the Gaussian Mixture Model with means, deviations, and weights of components.

        Parameters:
        means (list of numpy.ndarray): List of mean vectors of the components.
        deviations (list of numpy.ndarray): List of deviationsmatrices of the components.
        weights (list of float): List of weights of the components.

        Change to make this square: deviations: List of deviations. If x in [mean-deviation, mean+deviation], then it's an inliner.
        Does only allow inflate_full for now.
        """
        # for drawing inlines
        self.means = means
        self.deviations = deviations
        self.weights = weights
        assert np.allclose(sum(weights), 1), 'weights need to be sum up to 1'
        # for drawing outliers
        d = means[0].shape[0]  # Dimensionality of the data
        self.d = d
        self.border_extend_factor=border_extend_factor
        self.threshold_plus_delta = 1+delta
        self.threshold_minus_delta = 1-delta
        n = d if inflate_full else np.random.randint(1, d + 1)  # Generate random integer between 1 and d, inclusive
        if inflate_full and sub_dims is not None:
            print('we are inflating all the dimensions, however, sub_dims is provided')
            raise Exception
        self.sub_dims = np.sort(np.random.choice(d, size=n, replace=False)) if sub_dims is None else sub_dims

    def draw_samples(self, num_samples):
        """
        Draws samples from the 2D Gaussian Mixture Model.

        Parameters:
        num_samples (int): Number of samples to draw.

        Returns:
        numpy.ndarray: Array of samples drawn from the Gaussian Mixture Model.
        """
        samples = np.zeros(shape=(num_samples, self.d))
        component_choices = np.random.choice(len(self.weights), size=num_samples, p=self.weights)

        for cluster_id in range(len(self.weights)):
            center=self.means[cluster_id]
            deviation=self.deviations[cluster_id]
            minima, maxima=center-deviation, center+deviation

            cluster_sample_mask = (component_choices == cluster_id)
            cluster_samples = np.random.uniform(low=minima, high=maxima, size=(cluster_sample_mask.sum(), self.d))
            samples[cluster_sample_mask] = cluster_samples
        return samples

    def draw_inflated_samples(self, num_samples):
        """
        Draws samples from the 2D Gaussian Mixture Model.

        Parameters:
        num_samples (int): Number of samples to draw.

        Returns:
        numpy.ndarray: Array of samples drawn from the Gaussian Mixture Model.
        """
        samples = np.zeros(shape=(num_samples, self.d))
        component_choices = np.random.choice(len(self.weights), size=num_samples, p=self.weights)

        for cluster_id in range(len(self.weights)):
            center=self.means[cluster_id]
            deviation=self.deviations[cluster_id]*self.border_extend_factor
            minima, maxima=center-deviation, center+deviation
            cluster_sample_mask = (component_choices == cluster_id)
            cluster_samples = np.random.uniform(low=minima, high=maxima, size=(cluster_sample_mask.sum(), self.d))
            samples[cluster_sample_mask] = cluster_samples
        return samples

    @staticmethod
    def draw_samples_parallel(num_samples, num_workers=None, draw_fn=None):
        """
        Draws samples from the 2D Gaussian Mixture Model using multiprocessing.

        Parameters:
        num_samples (int): Total number of samples to draw.
        num_workers (int): Number of parallel workers (processes). Defaults to the number of CPU cores.

        Returns:
        numpy.ndarray: Array of samples drawn from the Gaussian Mixture Model.
        """
        # Determine the number of workers (default to the number of CPU cores)
        if num_workers is None:
            num_workers = cpu_count()

        # Split the total samples among workers
        samples_per_worker = num_samples // num_workers
        additional_samples = num_samples % num_workers

        # Create a list of sample sizes for each worker
        sample_sizes = [samples_per_worker] * num_workers

        # Adjust for any remaining samples
        if additional_samples > 0:
            sample_sizes[-1] += additional_samples

        # Use multiprocessing Pool to parallelize the sampling
        with Pool(num_workers) as pool:
            results = pool.map(draw_fn, sample_sizes)

        # Combine the results from all workers
        return np.vstack(results)

    def cubic_distance(self, sample, mean, deviation):
        """
        Computes the Mahalanobis distance of a sample from a given mean and inverse deviationsmatrix.

        Parameters:
        sample (numpy.ndarray): Sample point. (d, )
        mean (numpy.ndarray): Mean vector.  (d, )
        inv_deviations(numpy.ndarray): Inverse deviationsmatrix.  (sub-dims, )
        #change here: inv_deviations>deviation. Also calculate 1/deviation linf metric

        Returns:
        float: Mahalanobis distance of the sample from the mean.
        """
        delta=np.abs(sample[self.sub_dims]-mean[self.sub_dims])
        weight=1/(deviation[self.sub_dims]+0.0000001)
        return np.max(delta*weight)
        #return cubic(sample[self.sub_dims], mean[self.sub_dims], inv_deviations
        # the inv_deviationsis already the inverse of (sub-dim x sub-dim) deviations
    def batched_squared_cubic_distance(self, X, mean, deviation):
        # Subtract the mean from each input vector
        delta = X - mean  # (#samples, d-dimensional)
        delta = delta[:, self.sub_dims]  # (#samples, (sub-dims)-dimensional)
        weight=1/(deviation[self.sub_dims]+0.0000001) #(sub-dims, )
        distance=np.max(np.abs(delta)*weight,axis=-1)
        return distance

    def draw_inliners(self, num_samples):
        """
        Draws samples from the 2D Gaussian Mixture Model where points are within two sigma of any component.

        Parameters:
        num_samples (int): Number of samples to draw.

        Returns:
        numpy.ndarray: Array of samples drawn from the GMM that are within n-th percentile.
        """
        samples = []
        while len(samples) < num_samples:
            sample = self.draw_samples(1)[0]
            distances = [self.cubic_distance(sample, mean, deviation) for mean, deviation in
                         zip(self.means, self.deviations)]
            if np.min(distances) < self.threshold_minus_delta:
                samples.append(sample)
        return np.array(samples)

    def draw_local_anomalies(self, num_samples):
        """
        Draws samples from the 2D Gaussian Mixture Model where points are outside two sigma of any component.

        Parameters:
        num_samples (int): Number of samples to draw.

        Returns:
        numpy.ndarray: Array of samples drawn from the GMM that are outside n-th percentile.
        """
        samples = []
        while len(samples) < num_samples:
            sample = self.draw_inflated_samples(1)[0]
            distances = [self.cubic_distance(sample, mean, deviation) for mean, deviation in
                         zip(self.means, self.deviations)]
            if np.min(distances) > self.threshold_plus_delta:
                samples.append(sample)
        return np.array(samples)

    def assert_inliners(self, samples):
        for sample in samples:
            distances = [self.cubic_distance(sample, mean, deviation) for mean, deviation in
                         zip(self.means, self.deviations)]
            assert np.min(distances) < self.threshold_minus_delta

    def assert_local_anomalies(self, samples):
        for sample in samples:
            distances = [self.cubic_distance(sample, mean, deviation) for mean, deviation in
                         zip(self.means, self.deviations)]
            assert np.min(distances) > self.threshold_plus_delta

    def draw_batched_data(self, num_inliners, num_local_anomalies):
        raw_inliners = self.draw_samples(num_samples=int(num_inliners * 1.5))  # num_samples, d
        # ~80% of the num_inliners will be accepted, inflating the sample size to 150% gives us ~90%
        raw_local_anomalies = self.draw_inflated_samples(num_samples=int(num_local_anomalies * 2.0))
        #increased factor to 200% here

        # for similar reasons above

        def get_squared_batched_dist(raw_samples):
            batch_dist = []
            for mean, deviation in zip(self.means, self.deviations):
                distances = self.batched_squared_cubic_distance(X=raw_samples, mean=mean, deviation=deviation)
                batch_dist.append(distances)
            batch_dist = np.array(batch_dist).T
            return batch_dist

        inliners_squared_dist = get_squared_batched_dist(raw_samples=raw_inliners)  # num_raw_samples, num_cluster
        local_anomalies_squared_dist = get_squared_batched_dist(raw_samples=raw_local_anomalies)

        min_inliners_squared_dist = np.min(inliners_squared_dist, axis=-1)
        min_local_anomalies_squared_dist = np.min(local_anomalies_squared_dist, axis=-1)

        inliners_mask = (min_inliners_squared_dist < self.threshold_minus_delta)
        local_anomalies_mask = (min_local_anomalies_squared_dist > self.threshold_plus_delta)

        inliners = raw_inliners[inliners_mask]  # select those passed the criteria
        local_anomalies = raw_local_anomalies[local_anomalies_mask]

        inliners = inliners[:num_inliners]
        local_anomalies = local_anomalies[:num_local_anomalies]

        def add_extra(existing_samples, target_num_samples, draw_func):
            if existing_samples.shape[0] < target_num_samples:
                extra_samples = draw_func(num_samples=target_num_samples - existing_samples.shape[0])
                existing_samples = np.concatenate([existing_samples, extra_samples], axis=0)
            return existing_samples

        inliners = add_extra(existing_samples=inliners, target_num_samples=num_inliners, draw_func=self.draw_inliners)
        local_anomalies = add_extra(existing_samples=local_anomalies, target_num_samples=num_local_anomalies,
                                    draw_func=self.draw_local_anomalies)
        return inliners, local_anomalies

    def draw_batched_data_parallel(self, num_inliners, num_local_anomalies, num_workers):
        raw_inliners = self.draw_samples_parallel(num_samples=int(num_inliners * 1.5), num_workers=num_workers,
                                                  draw_fn=self.draw_samples)  # num_samples, d
        raw_local_anomalies = self.draw_samples_parallel(num_samples=int(num_local_anomalies * 1.1),
                                                         num_workers=num_workers, draw_fn=self.draw_inflated_samples)

        def get_squared_batched_dist(raw_samples):
            batch_dist = []
            for mean, deviation in zip(self.means, self.deviations):
                distances = self.batched_squared_cubic_distance(X=raw_samples, mean=mean, deviation=deviation)
                batch_dist.append(distances)
            batch_dist = np.array(batch_dist).T
            return batch_dist

        inliners_squared_dist = get_squared_batched_dist(raw_samples=raw_inliners)  # num_raw_samples, num_cluster
        local_anomalies_squared_dist = get_squared_batched_dist(raw_samples=raw_local_anomalies)

        min_inliners_squared_dist = np.min(inliners_squared_dist, axis=-1)
        min_local_anomalies_squared_dist = np.min(local_anomalies_squared_dist, axis=-1)

        inliners_mask = (min_inliners_squared_dist < self.threshold_minus_delta)
        local_anomalies_mask = (min_local_anomalies_squared_dist > self.threshold_plus_delta)

        inliners = raw_inliners[inliners_mask]  # select those passed the criteria
        local_anomalies = raw_local_anomalies[local_anomalies_mask]

        inliners = inliners[:num_inliners]
        local_anomalies = local_anomalies[:num_local_anomalies]

        def add_extra(existing_samples, target_num_samples, draw_func):
            if existing_samples.shape[0] < target_num_samples:
                extra_samples = draw_func(num_samples=target_num_samples - existing_samples.shape[0])
                existing_samples = np.concatenate([existing_samples, extra_samples], axis=0)
            return existing_samples

        inliners = add_extra(existing_samples=inliners, target_num_samples=num_inliners, draw_func=self.draw_inliners)
        local_anomalies = add_extra(existing_samples=local_anomalies, target_num_samples=num_local_anomalies,
                                    draw_func=self.draw_local_anomalies)
        return inliners, local_anomalies


def make_NdMclusterCMM(dim: int, num_cluster: int, weights: list[float], max_mean: int, min_dev:float, max_dev: float,
                       inflate_full: bool, sub_dims=None, percentile=0.80, delta=0.05):
    means = [np.random.random(dim) * np.random.randint(low=-max_mean, high=max_mean, size=dim)
             for _ in range(num_cluster)]
    deviations = [np.abs(np.random.random(dim) * np.random.uniform(low=min_dev, high=max_dev, size=dim))
                   for _ in range(num_cluster)]
    #deviations = []
    #for _ in range(num_cluster):
    #    diag = np.random.random(dim) * np.random.randint(low=1, high=max_var, size=dim)
    #    diag[diag == 0] = max_var / 2
    #    cov = np.diag(diag)
    #    deviations.append(cov)
    N_d_M_cluster_cubic = CubeMixtureModel(means=means, deviations=deviations, weights=weights,
                                                  inflate_full=inflate_full, sub_dims=sub_dims,
                                                  percentile=percentile, delta=delta)
    return N_d_M_cluster_cubic


def generate_constrained_eigenvals(d):
    # Generate uniformly distributed values in the range (-0.8, -0.2)
    low_range = np.random.uniform(-1.0, -0.1, size=d)

    # Generate uniformly distributed values in the range (0.2, 0.8)
    high_range = np.random.uniform(0.1, 1.0, size=d)

    # Randomly choose between the two ranges for each element
    choice = np.random.choice([0, 1], size=d)
    vector = np.where(choice == 0, low_range, high_range)

    return vector


def generate_full_rank_matrix(dim, scale=1):
    # Generate a random orthogonal matrix using QR decomposition
    A = np.random.rand(dim, dim)
    Q, _ = np.linalg.qr(A)

    eigenvals = generate_constrained_eigenvals(d=dim)
    eigenvals = np.diag(eigenvals)

    full_rank_matrix = Q @ eigenvals @ Q.T

    # print('inside gen full rank')
    assert np.linalg.matrix_rank(full_rank_matrix) == dim
    return full_rank_matrix


def generate_linear_transform(dim, A_scale=1, b_scale=1):
    A = generate_full_rank_matrix(dim=dim, scale=A_scale)
    # A = generate_full_rank_matrix_with_eigenvalues(dim=dim, scale=A_scale)
    b = np.random.rand(dim) * np.random.randint(low=-b_scale, high=b_scale + 1, size=dim)  # [low, high)
    return A, b


def transform_means(means, sub_dims, A, b):
    trans = []
    for mean in means:
        new_mean = mean.copy()
        new_mean[sub_dims] = A @ new_mean[sub_dims] + b
        trans.append(new_mean)
    return trans


def transform_covs(covs, sub_dims, A):
    trans = []
    for cov in covs:
        new_cov = cov.copy()
        new_cov[np.ix_(sub_dims, sub_dims)] = A @ new_cov[np.ix_(sub_dims, sub_dims)] @ A.T
        trans.append(new_cov)
    return trans


def transform_samples(samples, sub_dims, A, b):
    new_samples = samples.copy()
    new_samples[:, sub_dims] = new_samples[:, sub_dims] @ A.T + b

    return new_samples


if __name__ == "__main__":
    s = time.time()
    dim = np.random.randint(low=2, high=41)  # draw from [2, 20]
    num_cluster = np.random.randint(low=2, high=6)  # draw from [2, 5]
    max_mean = np.random.randint(low=2, high=6)  # draw from [2, 5]
    max_var = np.random.randint(low=2, high=6)  # draw from [2, 5]

    model = make_NdMclusterGMM(dim=dim, num_cluster=num_cluster, weights=[1 / num_cluster] * num_cluster,
                               max_mean=max_mean, max_var=max_var, inflate_full=False, sub_dims=None,
                               percentile=0.9, delta=0.05)

    num_samples = 5000

    print('testing the speed improvement of parallelization over seq-len')

    inliners, local_anomalies = model.draw_batched_data(num_inliners=num_samples, num_local_anomalies=num_samples)
    print(inliners.shape, local_anomalies.shape)
    print('w/o parallelization', time.time() - s)

    para_s = time.time()
    inliners, local_anomalies = model.draw_batched_data_parallel(num_inliners=num_samples, num_local_anomalies=num_samples,
                                                                 num_workers=10)
    print(inliners.shape, local_anomalies.shape)
    print('w. parallelization', time.time() - para_s)


    # test the batch data follows the inliner/local_anomaly criterion
    inliners, local_anomalies = model.draw_batched_data(num_samples, num_samples)
    model.assert_inliners(inliners)
    model.assert_local_anomalies(local_anomalies)

    # test whether linear transform preserves percentiles
    for _ in range(num_samples):
        full_rank_matrix = generate_full_rank_matrix(dim=dim)
        assert np.linalg.matrix_rank(full_rank_matrix) == dim
    print('full rank matrix generation asserted')

    sub_dims = model.sub_dims
    A, b = generate_linear_transform(dim=len(sub_dims))

    model_T = CubeMixtureModel(means=transform_means(model.means, sub_dims, A, b),
                                   deviations=transform_covs(model.deviations, sub_dims, A),
                                   weights=model.weights,
                                   sub_dims=sub_dims)

    in_T = transform_samples(inliners, sub_dims, A, b)
    la_T = transform_samples(local_anomalies, sub_dims, A, b)

    model_T.assert_inliners(in_T)  # linear transformed inliners remain within the n-th percentiles under new model
    model_T.assert_local_anomalies(la_T)  # similar to above
    print('linear transform successfully asserted')
