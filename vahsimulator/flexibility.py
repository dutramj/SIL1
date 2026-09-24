
# the modal data for each mode consist of the mode frequency (w_j), the generalized mass (mg_j), and generalized mode shapes (phi_jk) and slopes (sigma_jk), at different locations k on the vehicle (nodes).
# 20-80 modes are sufficient to create an accurate representation of the structural flexibility of the vehicle.

# A Flight Dynamics Model for a Multi-Actuated Flexible Rocket Vehicle
# Jeb S. Orr
# Science Applications International Corporation, Huntsville, AL 35806, USA

# r effectors

# each effector:
#     - mei: mass
#     - Jei: inertia tensor
#     - rei: displacement with respect to each Gi
#     - Gi: gimbal frame
#     - tei: set of effector torques
#     - rGi: displacement from the origin of B

# n linear oscilators (slosh mass)
# spring-mass damper system
#     - msj: mass
#     - rsj: umperturbed location with respect to B
#     - psj: local translation
#     - Jsj: inertia of the sloshing fluid (invariant)

# flexibility is in mass-normalized diagnonal form (orthogonal modes of a finite
# element model)
# p degrees of freedom
# η, η_dot: generalized displacement and generalized velocity -> dimension p
# Φk, Ψk: eigenvectors, corresponding to the translational and rotational
# components of the eigenvectors at each discrete gridpoint

