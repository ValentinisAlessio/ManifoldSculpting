import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class ManifoldSculpting:
    def __init__ (self, n_neighbors = 5, proj_dims = 2, niter = 100, sigma = 0.99, err=8e-2, apply_pca = True, patience = 150):
        '''
        Parameters:
        k: int
            Number of nearest neighbors to consider
        proj_dims: int
            Number of dimensions to project the data to
        niter: int
            Number of iterations
        sigma: float
            Scaling factor, set to 0.99 as in the paper
        apply_pca: bool
            Whether to apply PCA to the data
        patience: int
            Number of iterations to wait before stopping if convergence is not reached
        '''

        self.n_neighbors = n_neighbors
        self.proj_dims = proj_dims
        self.niter = niter
        self.sigma = sigma
        self.apply_pca = apply_pca

        self.err = err

        self.patience = patience

    def fit_transform(self, data):
        '''
        Fit the embedding of the data
        
        Parameters:
        data: np.array
            Data to fit the embedding

        Returns:
        np.array
            Best data after the embedding
        '''

        self.scale_factor = 1

        self.data = data
        # Find neighbours, distances, avg_distances, colinear points and angle for the data on which the ManifoldSculpting is applied
        self.neighbours, self.delta0, self.delta_avg, self.colinear, self.theta0 = self._compute_neighbourhood()
        self.learning_rate = self.delta_avg

        # Apply PCA to the data
        if self.apply_pca:
            self.pca_data = self._align_to_pc()
            self.d_preserved = np.arange(self.proj_dims, dtype=np.int32)
            self.d_scale = np.arange(self.proj_dims, self.data.shape[1], dtype=np.int32)

        else:
            cov = np.cov(self.data.T)
            most_important = np.argsort(-np.diag(cov)).astype(np.int32)
            self.d_pres = most_important[:self.n_dim]
            self.d_scal = most_important[self.n_dim:]
            self.pca_data = np.copy(self.data)

        # Adjust variables for a bunch of times to get to a reasonable starting point to compare errors
        epoch = 1
        with tqdm(total=np.inf, desc='Initial adjustment') as pbar:
            while self.scale_factor > 0.01:
                _ = self._step()
                epoch += 1
                pbar.update(1)

        epochs_since_improvement = 0
        best_error = np.inf

        # Adjust variables untill error plateaus or maximum iterations are reached
        with tqdm(total=self.niter, desc='Main loop') as pbar:
            pbar.update(epoch)
            while (epoch < self.niter) and (best_error > self.err) and (epochs_since_improvement < self.patience):
                
                mean_error = self._step()

                if mean_error < best_error:
                    best_error = mean_error
                    self.best_data = np.copy(self.pca_data)
                    self.best_error = best_error
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement == self.patience:
                        print('Convergence reached by patience')

                epoch += 1
                pbar.update(1)

        self.total_epochs = epoch
        self.last_error = mean_error

        return self.best_data

    def compute_error(self, p, visited):
        '''
        Compute the error for a point eps_pi
        
        Parameters:
        p: int
            Index of the point to compute the error for
        visited: np.array
            Indices of the points that have already been visited
            
        Returns:
        total_error: float
            Total error for the point
        '''

        # Initialize the weights
        weights = np.where(np.isin(self.neighbours[p], visited), 10, 1)

        # Calculate vectors `a` and `b`
        a = self.pca_data[p] - self.pca_data[self.neighbours[p]]
        b = self.pca_data[self.colinear[p]] - self.pca_data[self.neighbours[p]]

        # Compute norms `la` and `lb`
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)

        # Compute the angles `theta` between vectors `a` and `b`
        cos_theta = np.clip(np.sum(a * b, axis=1) / (la * lb), -1, 1)
        theta = np.arccos(cos_theta)

        # Compute error components
        err_dist = 0.5 * (la - self.delta0[p]) / self.delta_avg
        err_theta = (theta - self.theta0[p]) / np.pi

        # Compute total weighted error
        total_error = np.sum(weights * (err_dist**2 + err_theta**2))

        return total_error

    def _compute_neighbourhood(self):
        '''
        Find neighbours, distances, avg_distances, colinear points and angle for the data on which the ManifoldSculpting is applied
        
        Returns:
        tuple of np.arrays
            neighbours: indices of the k nearest neighbors for each point
            delta0: distances to the k nearest neighbors for each point
            avg_distances: average distance to the k nearest neighbors
            colinear: indices of the most colinear points for each point
            theta0: angles between the most colinear points for each point
        '''

        N = self.data.shape[0]

        # Find the k nearest neighbors
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors+1)
        neigh.fit(self.data)
        distances, indices = neigh.kneighbors(self.data)
        _dist, _ind = distances[:,1:], indices[:,1:]

        # Compute the average distance to the k nearest neighbors
        avg_distances = np.mean(_dist)

        # Find the most colinear points

        _colinear = np.zeros((N, self.n_neighbors), dtype=np.int32)
        _theta = np.zeros((N, self.n_neighbors), dtype=np.float32)

        # Could be nice to parallelize this
        for i in range(N):
            # for each point iterate over neighbours
            for nj,j in enumerate(_ind[i]):
                a = self.data[i] - self.data[j]
                la = np.linalg.norm(a)

                # compute and keep track of all angles between i-j-k, where
                # k are neighbours of j
                angles = np.zeros(self.n_neighbors)
                for nk,k in enumerate(_ind[j]):
                    b = self.data[k] - self.data[j]
                    lb = np.linalg.norm(b)
                    angles[nk] = np.arccos(np.minimum(1, np.maximum(np.dot(a,b)/(la*lb), -1)))

                # choose the point such that the angle is the closest to pi
                index = np.argmin(np.abs(angles-np.pi))
                _colinear[i,nj] = _ind[j,index]
                _theta[i,nj] = angles[index] 

        return _ind, _dist, avg_distances, _colinear, _theta

    def _align_to_pc(self):
        '''
        Align the data to the principal components. We assume to pass centered data.
        We don't need to reduce the dimensions, as it is already the scope of the main algorithm, we need only a traslation and a rotation.
        It is shown that leads to faster convergence.
        '''

        # Compute the covariance matrix
        cov = np.cov(self.data.T)

        # Compute the eigenvectors
        eigvals, eigvecs = np.linalg.eig(cov)

        # Sort the eigenvectors by decreasing eigenvalues
        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:,idx].astype(np.float32)

        return self.data @ eigvecs

    def _avg_neigh_dist(self):
        '''
        Compute the average distance to the heighbours for each point
        
        Returns:
        float
            Average distance to the neighbours
        '''

        diffs = self.pca_data[:, np.newaxis, :] - self.pca_data[self.neighbours]  # Shape: (N, k, dim)

        # Compute the norms of the differences
        distances = np.linalg.norm(diffs, axis=2)  # Shape: (N, k)
        
        # Compute the total distance and the count of distances
        total_distance = np.sum(distances)
        total_count = distances.size
        
        # Calculate the average distance
        average_distance = total_distance / total_count

        return average_distance

    def _adjust_point (self, p, visited):
        '''
        Adjust one point using a local Hill-Climbing algorithm
        
        Parameters:
        p: int
            Index of the point to adjust
        visited: np.array
            Indices of the points that have already been visited
            
        Returns:
        s: int
            Number of iterations
        err: float
            Error of the point
        '''

        lr = self.learning_rate * np.random.uniform(0.3, 1)
        improved = True

        err = self.compute_error(p, visited)
        s = 0

        while (s<30) and improved:
            s += 1
            improved = False

            for i in self.d_preserved:
                
                # Look only in the direction of the components to be preserved
                self.pca_data[p,i] += lr
                new_err = self.compute_error(p, visited)

                if new_err >= err:
                    self.pca_data[p,i] -= 2*lr
                    new_err = self.compute_error(p, visited)

                if new_err >= err:
                    self.pca_data[p,i] += lr
                    
                else:
                    err = new_err
                    improved = True
        
        return s-1, err

    def _step(self):
        '''
        Perform one step of the algorithm:
            - scale down dimensions to be removed
            - scale up dimensions the other dimensions and adjust points
            
        Returns:
        mean_error: float
            Mean error of the points
        '''

        # Pick the origin for the breadth-first adjustment
        origin = np.random.randint(self.data.shape[0], dtype=np.int32)

        q = []
        q.append(origin)
        visited = []

        self.scale_factor *= self.sigma

        # Scale down the dimensions to be removed
        self.pca_data[:,self.d_scale] *= self.sigma

        # Scale up preserved dimensions to keep the average distance
        # (not clear when this is stated in the paper, but in the original implementation it is present)
        while self._avg_neigh_dist() < self.delta_avg:
            self.pca_data[:,self.d_preserved] /= self.sigma

        step = 0
        mean_error = 0
        counter = 0 # Count the number of points that have been adjusted. Should be equal to the number of points in the end

        while q:
            p = q.pop(0)

            # Be sure not to visit the same point twice
            if p in visited:
                continue

            q.extend(self.neighbours[p])

            s, err = self._adjust_point(p, visited)
            step += s
            mean_error += err
            counter += 1
            visited.append(p)
            
        mean_error /= counter

        # If not many improvements, reduce lr
        # If many improvements, increase lr
        # NOTE: these are the values used in the author's implementation

        if step < self.pca_data.shape[0]:
            self.learning_rate *= 0.87
        else:
            self.learning_rate /= 0.91

        return mean_error