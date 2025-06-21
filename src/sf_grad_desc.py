#------------------------------------------------------------------------------
#   STREAMFUNCTION OPTIMIZATION - GRADIENT DESCENT METHOD
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def grad_x(f, dx):
    df_dx = np.zeros_like(f)
    df_dx[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dx)
    df_dx[0, :] = (f[1, :] - f[0, :]) / dx
    df_dx[-1, :] = (f[-1, :] - f[-2, :]) / dx
    return df_dx

def grad_y(f, dy):
    df_dy = np.zeros_like(f)
    df_dy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
    df_dy[:, 0] = (f[:, 1] - f[:, 0]) / dy
    df_dy[:, -1] = (f[:, -1] - f[:, -2]) / dy
    return df_dy

#------------------------------------------------------------------------------
# OPTIMIZATION FUNCTIONS
#------------------------------------------------------------------------------

def cmpt_gradanalytic(psi, ut, vt, dx, dy):
    
    uc = grad_y(psi, dy)
    vc = -grad_x(psi, dx)
    
    uerr = uc - ut
    verr = vc - vt
    
    Np = psi.shape[0] * psi.shape[1]
    au = uerr / Np
    av = verr / Np
    
    grad_u = -grad_y(au, dy)
    grad_v = grad_x(av, dx)
    
    grad = grad_u + grad_v
    
    return grad

def bc(psi):
    """Enforce zero boundary conditions"""
    psi_bc = psi.copy()
    psi_bc[0, :] = 0   # bottom boundary
    psi_bc[-1, :] = 0  # top boundary  
    psi_bc[:, 0] = 0   # left boundary
    psi_bc[:, -1] = 0  # right boundary
    return psi_bc

def zero_bcgrad(grad):
    """Zero out gradients at boundaries"""
    grad_bc = grad.copy()
    grad_bc[0, :] = 0   # bottom boundary
    grad_bc[-1, :] = 0  # top boundary  
    grad_bc[:, 0] = 0   # left boundary
    grad_bc[:, -1] = 0  # right boundary
    return grad_bc

#------------------------------------------------------------------------------
# GRADIENT DESCENT OPTIMIZER

def SGD_optimizer(psi_init, ut, vt, dx, dy, 
                              max_itr=5000, tol=1e-12, lr_init=1e-3,
                              lr_decay=None, verbose=True):
    """
    Simple gradient descent optimizer for streamfunction optimization
    
    Parameters:
    -----------
    psi_init : ndarray
        Initial streamfunction guess
    ut: ndarray
        Target u-velocity field
    vt : ndarray
        Target v-velocity field
    dx, dy : float
        Grid spacing in x and y directions
    max_itr : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    lr_init : float
        Learning rate
    lr_decay : float, optional
        Learning rate decay factor (lr *= lr_decay each iteration)
        If None, constant learning rate is used
    verbose : bool
        Print progress information
    
    Returns:
    --------
    psi : ndarray
        Optimized streamfunction
    ls : ndarray
        Loss history
    lrs : ndarray
        Learning rate history
    """
    
    psi = psi_init.copy()
    psi = bc(psi)
    
    # Learning rate
    lr = lr_init
    
    ls = []
    lrs = []
    
    if verbose:
        print("...Starting Gradient Descent optimization...")
        print(f"Parameters: lr={lr_init}", end="")
        if lr_decay is not None:
            print(f", decay={lr_decay}")
        else:
            print(" (constant)")
    
    for itr in range(max_itr):

        uc = grad_y(psi, dy)
        vc = -grad_x(psi, dx)
        uerr = uc - ut
        verr = vc - vt
        loss = 0.5 * np.mean(uerr**2 + verr**2)
        ls.append(loss)
        
        # Check for convergence
        if loss < tol:
            if verbose:
                print(f"Converged at iteration {itr} with loss {loss:.2e}")
            break
            
        grad = cmpt_gradanalytic(psi, ut, vt, dx, dy)
        grad = zero_bcgrad(grad)
        
        upd = lr * grad
        psi = psi - upd
        psi = bc(psi)
        
        lrs.append(lr)
        
        if lr_decay is not None:
            lr *= lr_decay
        
        if verbose and itr % 200 == 0:
            grad_norm = np.sqrt(np.mean(grad**2))
            print(f"Iter {itr}: Loss = {loss:.6e}, Grad norm = {grad_norm:.2e}, LR = {lr:.2e}")
    
    return psi, np.array(ls), np.array(lrs)

#------------------------------------------------------------------------------
# PROBLEM SETUP

def setup_test(N=100):
    """
    Set up a test problem with boundary-consistent streamfunction
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    
    Returns:
    --------
    X, Y : ndarray
        Coordinate meshgrids
    psit : ndarray
        True streamfunction (satisfies zero BCs)
    ut, vt : ndarray
        True velocity fields
    dx, dy : float
        Grid spacing
    """
    
    # Domain setup
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    
    pi = np.pi
    
    psit = np.sin(pi * X) * np.sin(pi * Y)
    ut = grad_y(psit, dy)
    vt = -grad_x(psit, dx)
    
    return X, Y, psit, ut, vt, dx, dy

def analyze(psi_opt, psit, ls, method_name="Gradient Descent"):
    """
    Analyze and print optimization results
    
    Parameters:
    -----------
    psi_opt : ndarray
        Optimized streamfunction
    psit : ndarray
        True streamfunction
    ls : ndarray
        Loss history
    method_name : str
        Name of optimization method
    """
    
    err = psit - psi_opt
    max_err = np.max(np.abs(err))
    rms_err = np.sqrt(np.mean(err**2))
    final_loss = ls[-1]
    
    print(f"\n{method_name} Optimization Results:")
    print("=" * 50)
    print(f"Final Loss:     {final_loss:.6e}")
    print(f"Max Error:      {max_err:.6e}")
    print(f"RMS Error:      {rms_err:.6e}")
    print(f"Iterations:     {len(ls)}")
    
    return {
        'final_loss': final_loss,
        'max_error': max_err,
        'rms_error': rms_err,
        'iterations': len(ls)
    }

def plot_results(X, Y, psit, psi_opt, ls, lrs):

    err = psit - psi_opt
    max_err = np.max(np.abs(err))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    cs1 = axes[0,0].contourf(X, Y, psit, levels=20, cmap='viridis')
    axes[0,0].contour(X, Y, psit, levels=10, colors='k', linewidths=0.5)
    axes[0,0].set_title('True Solution', fontsize=14)
    axes[0,0].set_xlabel(r'$x$', fontsize=12)
    axes[0,0].set_ylabel(r'$y$', fontsize=12)
    axes[0,0].set_aspect('equal')
    plt.colorbar(cs1, ax=axes[0,0])
    
    cs2 = axes[0,1].contourf(X, Y, psi_opt, levels=20, cmap='viridis')
    axes[0,1].contour(X, Y, psi_opt, levels=10, colors='k', linewidths=0.5)
    axes[0,1].set_title('Gradient Descent Solution', fontsize=14)
    axes[0,1].set_xlabel(r'$x$', fontsize=12)
    axes[0,1].set_ylabel(r'$y$', fontsize=12)
    axes[0,1].set_aspect('equal')
    plt.colorbar(cs2, ax=axes[0,1])
    
    cs3 = axes[1,0].contourf(X, Y, err, levels=20, cmap='RdBu_r', 
                             vmin=-max_err, vmax=max_err)
    axes[1,0].contour(X, Y, err, levels=8, colors='k', linewidths=0.5)
    axes[1,0].set_title(f'Error Field (max = {max_err:.2e})', fontsize=14)
    axes[1,0].set_xlabel(r'$x$', fontsize=12)
    axes[1,0].set_ylabel(r'$y$', fontsize=12)
    axes[1,0].set_aspect('equal')
    plt.colorbar(cs3, ax=axes[1,0])
    
    # Convergence history
    axes[1,1].semilogy(ls, 'g-', linewidth=2, label='Loss')
    axes[1,1].set_xlabel('Iteration', fontsize=12)
    axes[1,1].set_ylabel('Loss', fontsize=12)
    axes[1,1].set_title('Gradient Descent Convergence', fontsize=14)
    axes[1,1].grid(True, alpha=0.3)
    
    if len(set(lrs)) > 1:
        ax2 = axes[1,1].twinx()
        ax2.plot(lrs, 'orange', alpha=0.7, linewidth=1, linestyle='--', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Combine legends
        lines1, labels1 = axes[1,1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1,1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('SGD_opt_results.jpg', dpi=400, bbox_inches='tight')
    plt.show()
    
    # Convergence plot alone
    plt.figure(figsize=(10, 6))
    plt.semilogy(ls, 'g-', linewidth=2)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('SDG Optimizer Convergence', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('SGD_convergence.jpg', dpi=400, bbox_inches='tight')
    plt.show()

#------------------------------------------------------------------------------

def main():
    
    print("="*70)
    print("STREAMFUNCTION OPTIMIZATION WITH GRADIENT DESCENT METHOD")
    print("="*70)
    
    # Problem setup
    N = 100
    X, Y, psit, ut, vt, dx, dy = setup_test(N)
    
    print(f"Problem setup:")
    print(f"Grid size: {N}x{N}")
    print(f"Domain: [0,1] x [0,1]")
    print(f"True psi range: [{np.min(psit):.3f}, {np.max(psit):.3f}]")
    print(f"True u range: [{np.min(ut):.3f}, {np.max(ut):.3f}]")
    print(f"True v range: [{np.min(vt):.3f}, {np.max(vt):.3f}]")
    
    # Initial guess
    np.random.seed(42)
    psi_init = np.zeros((N, N))
    psi_init = bc(psi_init)
    
    print(f"\n...Running Gradient Descent optimization...")
    psi_opt, ls, lrs = SGD_optimizer(
        psi_init, ut, vt, dx, dy,
        max_itr=5000, tol=1e-12, lr_init=1e-3,
        lr_decay=None, verbose=True
    )
    
    error_analysis = analyze(psi_opt, psit, ls, "Gradient Descent")
    
    plot_results(X, Y, psit, psi_opt, ls, lrs)
    
    return psi_opt, psit, ls, error_analysis

if __name__ == "__main__":
    psi_opt, psit, ls, analysis = main()
