#used https://pennylane.ai/qml/demos/tutorial_vqe 

from jax import numpy as np
import jax
import optax
import pennylane as qml
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

# Load minimal hydrogen Hamiltonian
dataset = qml.data.load('qchem', molname="H2")[0]
H, qubits = dataset.hamiltonian, len(dataset.hamiltonian.wires)

dev = qml.device("lightning.qubit", wires=qubits)

# Hartree-Fock reference state: |1100>
electrons = 2
hf = qml.qchem.hf_state(electrons, qubits)

@qml.qnode(dev, interface="jax")
def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.expval(H)

def cost_fn(param):
    return circuit(param, wires=range(qubits))

# Optimizer setup
max_iterations = 100
conv_tol = 1e-6
opt = optax.sgd(learning_rate=0.4)

theta = np.array(0.0)
energy = [cost_fn(theta)]
angle = [theta]
opt_state = opt.init(theta)

for n in range(max_iterations):
    grad = jax.grad(cost_fn)(theta)
    updates, opt_state = opt.update(grad, opt_state)
    theta = optax.apply_updates(theta, updates)

    energy.append(cost_fn(theta))
    angle.append(theta)

    conv = np.abs(energy[-1] - energy[-2])
    if n % 2 == 0:
        print(f"Step {n},  Energy = {energy[-1]:.8f} Ha")
    if conv <= conv_tol:
        break

print(f"\n✅ Final energy = {energy[-1]:.8f} Ha")
print(f"✅ Optimal θ = {theta:.4f}")

# Plot
E_fci = -1.136189454088
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(energy, "go--")
plt.axhline(E_fci, color="red", linestyle="--", label="Exact FCI")
plt.title("Energy vs Optimization Step")
plt.xlabel("Step")
plt.ylabel("Energy (Ha)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(angle, "bo--")
plt.title("θ vs Optimization Step")
plt.xlabel("Step")
plt.ylabel("θ (radians)")

plt.tight_layout()
plt.show()
