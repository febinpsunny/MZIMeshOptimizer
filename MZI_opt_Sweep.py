import numpy as np
from scipy.optimize import minimize
from pyDOE import lhs
from tqdm import tqdm
from functools import partial
import random
import matplotlib.pyplot as plt

def matrixRead_util(filename):
    with open(filename, 'r') as file:
        # Read lines, strip any extra whitespace, and split elements into rows
        matrix = [line.strip().split() for line in file]
        
        # Convert each row to floats and then to a numpy array
        matrix = np.array(matrix, dtype=float)
    
    return matrix

# Fidelity function
def fidelityFunc(T, T_tilde, N):
    T = np.array(T)
    T_tilde = np.array(T_tilde)
    T_tilde_dagger = np.conjugate(T_tilde).T
    trace_value = np.trace(T_tilde_dagger @ T)
    fidelity_value = (np.abs(trace_value / N)) ** 2
    return fidelity_value

# Theta layer function
def theta_layer(n, theta):
    M = np.eye(n, dtype=complex)
    for ii in range(n):
        M[ii, ii] = np.exp(-1j * theta[ii])
    return M

# DC layer functions for odd and even indices
def DC_layer_odd(n, k):
    ids = [num for num in range(1, n + 1) if num % 2 != 0]
    if n in ids:
        ids.remove(n)
    mat = np.eye(n, dtype=complex)
    for h in range(len(ids)):
        T11 = np.sqrt(1 - k[h])
        T12 = 1j * np.sqrt(k[h])
        T21 = 1j * np.sqrt(k[h])
        T22 = np.sqrt(1 - k[h])
        mat[ids[h] - 1, ids[h] - 1] = T11
        if ids[h] < n:
            mat[ids[h] - 1, ids[h]] = T12
            mat[ids[h], ids[h] - 1] = T21
            mat[ids[h], ids[h]] = T22
    return mat

def DC_layer_even(n, k):
    ids = [num for num in range(1, n + 1) if num % 2 == 0]
    if n in ids:
        ids.remove(n)
    mat = np.eye(n, dtype=complex)
    for h in range(len(ids)):
        T11 = np.sqrt(1 - k[h])
        T12 = 1j * np.sqrt(k[h])
        T21 = 1j * np.sqrt(k[h])
        T22 = np.sqrt(1 - k[h])
        mat[ids[h] - 1, ids[h] - 1] = T11
        if ids[h] < n:
            mat[ids[h] - 1, ids[h]] = T12
            mat[ids[h], ids[h] - 1] = T21
            mat[ids[h], ids[h]] = T22
    return mat

# Super mesh function to compute transfer matrix
def super_mesh(n, n_even, n_odd, k_even, k_odd, n_theta, theta_init):
    temp_even = np.zeros((n, n, n_even + n_odd), dtype=complex)
    temp_odd = np.zeros((n, n, n_even + n_odd), dtype=complex)
    for kk in range(n_even + n_odd):
        temp_even[:, :, kk] = np.eye(n, dtype=complex)
        temp_odd[:, :, kk] = np.eye(n, dtype=complex)
    temp_theta = np.zeros((n, n, n_theta), dtype=complex)
    for jj in range(n_theta):
        temp_theta[:, :, jj] = theta_layer(n, theta_init[jj])
    for i in range(n_even):
        temp_even[:, :, i] = DC_layer_even(n, k_even[i])
    for ii in range(n_odd):
        temp_odd[:, :, ii] = DC_layer_odd(n, k_odd[ii])
    result = np.matmul(temp_odd[:, :, (n_even + n_odd - 1)], temp_even[:, :, (n_even + n_odd - 1)])
    for ii in range((n_even + n_odd - 2), -1, -1):
        result = np.matmul(result, np.matmul(temp_odd[:, :, ii], temp_even[:, :, ii]))
    Mat = np.matmul(result, temp_theta[:, :, 0])
    return Mat

# Cost function
def cost_function(params, n, n_even, n_odd, kappa_initial, theta_initial, T1, alpha=10.0, beta=0.01, gamma=0.1, drop_penalty_weight=5.0):
    # Initialize previous kappa, theta, and fidelity on the first call
    if not hasattr(cost_function, "kappa_previous"):
        cost_function.kappa_previous = kappa_initial
        cost_function.theta_previous = theta_initial
        cost_function.fidelity_previous = 0  # Set an initial low fidelity value

    # Compute `kappa_current` and `theta_current`
    num_kappa = len(cost_function.kappa_previous.flatten())
    kappa_current = np.reshape(params[:num_kappa], cost_function.kappa_previous.shape)
    theta_current = params[num_kappa:]
    theta_wrapped = np.array([theta_current])
    
    # Calculate transfer matrix using super mesh
    T = super_mesh(n, n_even, n_odd, kappa_current[:n_even], kappa_current[n_even:], 1, theta_wrapped)
    
    # Fidelity term
    fidelity = fidelityFunc(T1, T, n)
    fidelity_penalty = (1 / fidelity) ** 2  # Sharper penalty for lower fidelity

    # Difference from previous values (normalized) for change penalty
    kappa_diff = np.sum((kappa_current - cost_function.kappa_previous)**2) / kappa_current.size
    theta_diff = np.sum((theta_current - cost_function.theta_previous)**2) / theta_current.size
    change_penalty = kappa_diff + theta_diff

    # Penalty for decreased fidelity
    fidelity_drop_penalty = 0
    # if fidelity < cost_function.fidelity_previous:
    #     fidelity_drop_penalty = drop_penalty_weight * (1/(cost_function.fidelity_previous - fidelity))

    # Unitarity penalty with exponential scaling
    unitarity_penalty = np.linalg.norm(np.matmul(T, T.conj().T) - np.eye(n), 'fro')**2
    unitarity_penalty = np.exp(unitarity_penalty)  # Stronger penalty for deviations

    # Dynamic weighting (optional)
    # For example, lower alpha as fidelity improves beyond a threshold
    if fidelity > 0.6:
        alpha = alpha * 0.5

    # Update previous values
    cost_function.kappa_previous = np.copy(kappa_current)
    cost_function.theta_previous = np.copy(theta_current)
    cost_function.fidelity_previous = fidelity  # Store current fidelity as previous fidelity for the next iteration

    # Total cost with an additional penalty for fidelity drop
    return alpha * fidelity_penalty + beta * change_penalty + gamma * unitarity_penalty + fidelity_drop_penalty

def reset_cost_function():
    if hasattr(cost_function, "kappa_previous"):
        del cost_function.kappa_previous
    if hasattr(cost_function, "theta_previous"):
        del cost_function.theta_previous
    if hasattr(cost_function, "fidelity_previous"):
        del cost_function.fidelity_previous


def initialize_plot(title, xlabel, ylabel, color="blue"):
    """Initialize a live plot for a single metric and return figure, axis, and line objects."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], label=ylabel, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.draw()
    return fig, ax, line

def update_plot(line, ax, data, event_indices=None, event_color="red", event_label="Event"):
    """
    Update the live plot with new data and optional events.
    
    Parameters:
    - line: The line object to update.
    - ax: The axis object for adjustments.
    - data: List of metric values to plot.
    - event_indices: List of indices where events occurred.
    - event_color: Color for event markers.
    - event_label: Label for the first event marker.
    """
    # Update main line plot
    line.set_xdata(np.arange(len(data)))
    line.set_ydata(data)
    ax.set_xlim(0, len(data))
    ax.set_ylim(0, max(data) * 1.1 if max(data) > 0 else 1)

    # Plot event markers if any
    if event_indices:
        for idx in event_indices:
            ax.scatter(idx, data[idx], color=event_color, s=15, label=event_label if idx == event_indices[0] else "")

    plt.pause(0.01)

# Fidelity function for simulated annealing (calculate fidelity inline)
def calculate_fidelity(params, T1,  n_even, kappa_size):
    # Split `params` into `kappa` and `theta`
    _, n=T1.shape
    kappa = np.reshape(params[:kappa_size],(2*n_even,n // 2))       # kappa from 0 to kappa_size
    theta = params[kappa_size:]       # theta from kappa_size to the end
    
    # Reshape `kappa` to ensure correct dimensions for `n_even` and `n_odd`
    kappa_even = kappa[:n_even]
    kappa_odd = kappa[n_even:]

    # Wrap `theta` to match the expected structure for `super_mesh`
    theta_wrapped = np.array([theta])  # Wrap theta as a 2D array with shape (1, n)
    
    # Generate the transfer matrix T using `super_mesh`
    T = super_mesh(n, n_even, n_odd, kappa_even, kappa_odd, 1, theta_wrapped)
    
    # Calculate fidelity between T1 and the generated T
    return fidelityFunc(T1, T, n)

def compute_rvd(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same dimensions.")
    
    # Compute the Frobenius norm of (A - B)
    numerator = np.linalg.norm(A - B, ord='fro')
    
    # Compute the Frobenius norm of A
    denominator = np.linalg.norm(A, ord='fro')
    
    # Calculate RVD
    rvd = numerator / denominator
    return rvd

# Simulated Annealing without fidelity_function in the parameter list
def custom_simulated_annealing(cost_function, bounds, initial_solution, T1, n_even, kappa_size, initial_temp=1000, alpha=0.95, maxiter=5000, restart_count=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    progress_bar = tqdm(total=maxiter, desc="Simulated Annealing Progress")
    
    # Initialize variables
    best_solution = initial_solution.copy()
    best_cost = cost_function(best_solution)
    current_solution = best_solution.copy()
    current_cost = best_cost
    current_fidelity = calculate_fidelity(current_solution, T1,  n_even, kappa_size)
    temperature = initial_temp
    no_improvement_iters = 0

    # Track histories for each metric
    cost_history, temp_history, fidelity_history = [], [], []
    best_solution_updates, restarts = [], []

    # Unpack bounds into arrays for easier handling
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    bound_ranges = upper_bounds - lower_bounds

    # Optimization Loop
    for i in range(maxiter):
        # Track each metric's history
        cost_history.append(current_cost)
        temp_history.append(temperature)
        fidelity_history.append(current_fidelity)

        # Generate candidate solution
        normalized_current = (current_solution - lower_bounds) / bound_ranges
        perturbation = np.random.uniform(-1, 1, len(current_solution)) * (temperature / initial_temp)
        normalized_candidate = normalized_current + perturbation
        candidate = lower_bounds + normalized_candidate * bound_ranges
        candidate = np.clip(candidate, lower_bounds, upper_bounds)

        # Evaluate candidate
        candidate_cost = cost_function(candidate)
        candidate_fidelity = calculate_fidelity(candidate,  T1,  n_even, kappa_size)

        # Determine acceptance
        delta_cost = candidate_cost - current_cost
        acceptance_probability = np.exp(-delta_cost / temperature) if delta_cost > 0 else 1.0

        if np.random.rand() < acceptance_probability:
            current_solution, current_cost, current_fidelity = candidate, candidate_cost, candidate_fidelity

        # Track best solution
        if current_cost < best_cost:
            best_solution, best_cost = current_solution, current_cost
            no_improvement_iters = 0
            best_solution_updates.append(i)

        else:
            no_improvement_iters += 1

        # Decay temperature
        temperature *= alpha

        # Restart temperature if no improvement for many iterations
        if no_improvement_iters > restart_count:
            temperature = initial_temp
            no_improvement_iters = 0
            restarts.append(i)

        # Adaptive decay rate adjustment
        if no_improvement_iters > 200:
            alpha = min(0.999, alpha + 0.001)

        progress_bar.update(1)

    # Finalize plots and close progress bar
    progress_bar.close()

    return best_solution, best_cost, cost_history


# Utility function to plot cost history
def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label="Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function Across Simulated Annealing Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()

def optimize_hybrid_SA(n, n_even, n_odd, kappa_initial, theta_initial, T1, alpha=10.0, beta=0.01, gamma=0.1):
    bounds = [(0.0001, 0.9999)] * len(kappa_initial.flatten()) + [(0, 2 * np.pi)] * len(theta_initial.flatten())

    # Flatten and concatenate initial kappa and theta values
    initial_solution = np.concatenate([kappa_initial.flatten(), theta_initial.flatten()])

    # Prepare the cost function with additional parameters fixed
    sa_cost_func = partial(cost_function, n=n, n_even=n_even, n_odd=n_odd, kappa_initial=kappa_initial, theta_initial=theta_initial, T1=T1, alpha=alpha, beta=beta, gamma=gamma)

    # Step 1: Global Optimization with Simulated Annealing, starting from the initial solution
    best_solution, best_cost, cost_history = custom_simulated_annealing(
        sa_cost_func,
        bounds,
        initial_solution=initial_solution,
        T1=T1,
        n_even=n_even,
        kappa_size=len(kappa_initial.flatten()),
        initial_temp=20000,
        alpha=0.95,
        maxiter=20000,
        restart_count=1000,
        seed=42
    )

    best_solution = np.clip(best_solution, [b[0] for b in bounds], [b[1] for b in bounds])

    # Step 2: Local Optimization with L-BFGS-B (Refinement)
    def lbfgsb_cost_function(params):
        return cost_function(params, n, n_even, n_odd, kappa_initial, theta_initial, T1, alpha, beta, gamma)

    # Initialize loop parameters
    max_retries = 100  # Maximum number of retries
    target_cost = 9  # Target cost threshold
    current_cost = np.inf  # Initialize with a high cost
    retries = 0  # Track the number of retries
    success=0

    # Start with best_solution from previous optimization (e.g., simulated annealing)
    current_solution = best_solution

    # Repeat L-BFGS-B optimization until conditions are met
    while current_cost > target_cost and retries < max_retries:

        try:
        # First attempt with strict tolerance
            result_lbfgsb = minimize(
                    lbfgsb_cost_function,
                    current_solution,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True}
                )
        except ValueError as e:
            print("ValueError encountered during L-BFGS-B with strict ftol. Retrying with relaxed ftol...")
            print(f"Error Details: {e}")
        
            # Retry with relaxed tolerance
            result_lbfgsb = minimize(
                lbfgsb_cost_function,
                current_solution,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True}
            )

        # Update current cost and solution
        current_cost = result_lbfgsb.fun
        current_solution = np.clip(result_lbfgsb.x, [b[0] for b in bounds], [b[1] for b in bounds])
        current_solution = result_lbfgsb.x
        retries += 1  # Increment retry counter

        print(f"Attempt {retries}: Cost = {current_cost}")

        if result_lbfgsb.success:
            success=success+1
            if success>=(max_retries/20):
                print("Optimization converged successfully.")
                break
        else:
            success=0

    # Extract optimized kappa and theta
    optimized_params = result_lbfgsb.x
    num_kappa = len(kappa_initial.flatten())
    kappa_optimized = optimized_params[:num_kappa].reshape(kappa_initial.shape)
    theta_optimized = optimized_params[num_kappa:]
    return kappa_optimized, theta_optimized, cost_history

def initialize_kappa_theta(n_even, n_odd, n, target_kappa=0.5, kappa_std=0.1, theta_std=np.pi / 10):
    # Total size for kappa and theta initializations
    kappa_size = (n_even + n_odd) * (n // 2)
    
    # Hybrid approach: Start with LHS for broad coverage
    lhs_samples = lhs(kappa_size + n, samples=1)

    # Kappa initialization: Centered around target_kappa with Gaussian perturbation
    kappa_initial = lhs_samples[:, :kappa_size].reshape(n_even + n_odd, n // 2)
    kappa_initial = kappa_initial + kappa_std * np.random.randn(n_even + n_odd, n // 2)  # Gaussian centered at target_kappa
    kappa_initial = np.clip(kappa_initial, 0.0001, 0.9999)  # Ensure bounds are respected

    # Theta initialization: Start around zero with small Gaussian perturbation
    theta_initial = lhs_samples[:, kappa_size:] * (2 * np.pi)  # Uniform LHS sampling
    theta_initial = theta_initial + theta_std * np.random.randn(n)  # Small Gaussian perturbation
    theta_initial = np.clip(theta_initial, 0, 2 * np.pi)  # Ensure bounds for theta

    return kappa_initial, theta_initial

# Main code block
if __name__ == "__main__":
    # Parameters setup
    T_size_list=[16,32]
    for T_size in T_size_list:
        filename = "C:/Users/psfeb/Downloads/Amin/N"+str(T_size)+"_guass_without_loss_XT_real.txt"
        filename_i = "C:/Users/psfeb/Downloads/Amin/N"+str(T_size)+"_guass_without_loss_XT_imag.txt"

        for meshSize in [i * T_size for i in range(1, 6)]:
            reset_cost_function()
            n_even = meshSize
            n_odd = meshSize

            mat_real = matrixRead_util(filename)
            mat_imag = matrixRead_util(filename_i)
            T1 = mat_real + 1j * mat_imag
            _, n = T1.shape

            # Latin Hypercube Sampling for initial guesses
            kappa_initial, theta_initial = initialize_kappa_theta(n_even, n_odd, n)

            # Perform hybrid optimization
            kappa_optimized, theta_optimized, cost_history = optimize_hybrid_SA(n, n_even, n_odd, kappa_initial, theta_initial, T1)

            # Final calculation with optimized parameters
            theta_optimized_reshaped = np.array([theta_optimized])
            test1 = super_mesh(n, n_even, n_odd, kappa_optimized[:n_even], kappa_optimized[n_even:], 1, theta_optimized_reshaped)

            # Calculate final fidelity and Euclidean error
            total_distance = np.linalg.norm(T1 - test1)
            fidelity = fidelityFunc(T1, test1, n)
            rvd = compute_rvd(T1, test1)

            with open("C:/Users/psfeb/Downloads/Amin/N_"+str(T_size)+"_"+str(meshSize)+"_out.txt","w") as outFile:
                outFile.write(f"Optimized kappa values:\n{kappa_optimized}\n")
                outFile.write(f"Optimized theta values:\n{theta_optimized}\n")
                outFile.write(f"Output matrix (T1 @ ones):\n{np.matmul(T1, np.ones((n, 1)))}\n")
                outFile.write(f"Output matrix from optimized kappa and theta (test1 @ ones):\n{np.matmul(test1, np.ones((n, 1)))}\n")
                outFile.write(f"Final Euclidean Error: {total_distance}\n")
                outFile.write(f"Fidelity between the matrices: {fidelity}\n")
                outFile.write(f"RVD between matrices: {rvd}\n")

            #plot_cost_history(cost_history)