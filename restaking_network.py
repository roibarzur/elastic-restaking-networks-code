import math
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import milp, Bounds, LinearConstraint


class RestakingNetwork:
    service_names: npt.NDArray[np.str_]
    validator_names: npt.NDArray[np.str_]

    service_attack_rewards: npt.NDArray
    service_attack_thresholds: npt.NDArray
    validator_stakes: npt.NDArray
    allocations: npt.NDArray

    def __init__(self,
                 service_attack_rewards: npt.NDArray,
                 service_attack_thresholds: npt.NDArray,
                 validator_stakes: npt.NDArray,
                 allocations: npt.NDArray,
                 service_names: Optional[npt.NDArray[np.str_]] = None,
                 validator_names: Optional[npt.NDArray[np.str_]] = None
                 ):
        self.service_attack_rewards = service_attack_rewards.astype(np.float64)
        self.service_attack_thresholds = service_attack_thresholds.astype(np.float64)
        self.validator_stakes = validator_stakes.astype(np.float64)
        self.allocations = allocations.astype(np.float64)

        self.service_names = service_names if service_names is not None else np.array([f"S{i}" for i in range(len(self.service_attack_rewards))])
        self.validator_names = validator_names if validator_names is not None else np.array([f"V{i}" for i in range(len(self.validator_stakes))])

        assert self.validator_stakes.ndim == 1, "Validator stakes must be a 1D array"
        assert self.service_attack_rewards.ndim == 1, "Service attack rewards must be a 1D array"
        assert self.service_attack_thresholds.ndim == 1, "Service attack thresholds must be a 1D array"
        assert self.allocations.shape == (len(self.validator_stakes), len(self.service_attack_rewards)), "Allocations must be a 2D array with shape (len(validators), len(services))"

        if np.any(self.allocations):
            assert np.all(np.max(self.allocations, axis=1) <= self.validator_stakes), "Allocations must be less than or equal to validators' stakes"

        assert len(self.service_names) == len(self.service_attack_rewards), "Service names must have the same length as the number of services"
        assert len(self.service_attack_thresholds) == len(self.service_attack_rewards), "Service attack thresholds must have the same length as the number of services"
        assert len(self.validator_names) == len(self.validator_stakes), "Validator names must have the same length as the number of validators"

    def apply_byzantine_services(self, byzantine_services: npt.NDArray[np.bool_]) -> "RestakingNetwork":
        """
        Apply the byzantine services to the network.
        """
        remaining_service_names = np.array([self.service_names[j] for j in range(len(self.service_names)) if not byzantine_services[j]])
        remaining_service_attack_rewards = np.array([self.service_attack_rewards[j] for j in range(len(self.service_attack_rewards)) if not byzantine_services[j]])
        remaining_service_attack_thresholds = np.array([self.service_attack_thresholds[j] for j in range(len(self.service_attack_thresholds)) if not byzantine_services[j]])

        remaining_stakes = np.array([max(0, self.validator_stakes[i] - np.sum(self.allocations[i, byzantine_services.astype(bool)])) for i in range(len(self.validator_stakes))])
        remaining_allocations = np.minimum(np.array([self.allocations[i, ~byzantine_services.astype(bool)] for i in range(len(self.validator_stakes))]), remaining_stakes.reshape(-1, 1))

        return RestakingNetwork(
            service_attack_rewards=remaining_service_attack_rewards,
            service_attack_thresholds=remaining_service_attack_thresholds,
            validator_stakes=remaining_stakes,
            allocations=remaining_allocations,
            service_names=remaining_service_names,
            validator_names=self.validator_names
        )

    def check_security_with_milp(self) -> tuple[bool, Optional[float], Optional[npt.NDArray[np.bool]], Optional[tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]]]:
        """
        Check if the network is secure using scipy's optimization.
        
        Returns:
            Tuple of:
            - is_secure: Whether the network is secure
            - profit: The profit of the network
            - attacked_services: The services that are attacked
            - attack_stakes: The stakes used to attack the services
        """
        n_validators = len(self.validator_stakes)
        n_services = len(self.service_attack_rewards)

        # Variables:
        # x_j^S: Binary variable indicating if service j is attacked (milpAttackedServiceVariable)
        # x_ij^α: Amount of stake validator i uses to attack service j (milpAttackStakeVariable) 
        # x_i^c: Cost for validator i (milpValidatorCostVariable)
        # x_i^{c,aux}: Auxiliary binary variable for validator i cost calculation (milpValidatorCostAuxiliaryVariable)

        # Variable indices in the solution vector x:
        idx_x_j_S = lambda j: j  # x_j^S starts at 0
        idx_x_ij_alpha = lambda i, j: n_services + i*n_services + j  # x_ij^α starts after x_j^S
        idx_x_i_c = lambda i: n_services + n_validators*n_services + i  # x_i^c starts after x_ij^α
        idx_x_i_c_aux = lambda i: n_services + n_validators*n_services + n_validators + i  # x_i^{c,aux} starts after x_i^c

        n_vars = n_services + n_validators * n_services + n_validators + n_validators

        # Objective: Maximize sum(R_j * x_j^S) - sum(x_i^c)
        c = np.zeros(n_vars)
        for j in range(n_services):
            c[idx_x_j_S(j)] = self.service_attack_rewards[j]  # Coefficient for x_j^S
        for i in range(n_validators):
            c[idx_x_i_c(i)] = -1  # Coefficient for x_i^c

        # Constraints matrix
        constraints = []
        b_ub = []

        # At least one service must be attacked: sum(x_j^S) >= 1
        row = np.zeros(n_vars)
        for j in range(n_services):
            row[idx_x_j_S(j)] = -1
        constraints.append(row)
        b_ub.append(-1)

        # Attack feasibility constraints for each service:
        # sum(x_ij^α) >= θ_j * sum(w_ij) - M_feasibility * (1 - x_j^S)
        M_feasibility = max(self.service_attack_thresholds[j] * np.sum(self.allocations[:, j]) for j in range(n_services))
        for j in range(n_services):
            threshold = self.service_attack_thresholds[j] * np.sum(self.allocations[:, j])
            row = np.zeros(n_vars)
            for i in range(n_validators):
                row[idx_x_ij_alpha(i, j)] = -1  # Coefficient for x_ij^α
            row[idx_x_j_S(j)] = M_feasibility  # Coefficient for x_j^S
            constraints.append(row)
            b_ub.append(M_feasibility - threshold)

        # Calculate M_cost for big-M constraints
        M_cost = max(np.max(self.validator_stakes), np.max(np.sum(self.allocations, axis=1)))

        # Validator cost constraints with auxiliary variables
        for i in range(n_validators):
            # x_i^c <= sum(x_ij^α)
            row = np.zeros(n_vars)
            row[idx_x_i_c(i)] = 1
            for j in range(n_services):
                row[idx_x_ij_alpha(i, j)] = -1
            constraints.append(row)
            b_ub.append(0)

            # x_i^c >= stake_i - M_cost * z_i
            row = np.zeros(n_vars)
            row[idx_x_i_c(i)] = -1
            row[idx_x_i_c_aux(i)] = -M_cost
            constraints.append(row)
            b_ub.append(-self.validator_stakes[i])

            # x_i^c >= sum(x_ij^α) - M_cost * (1-z_i)
            row = np.zeros(n_vars)
            row[idx_x_i_c(i)] = -1
            for j in range(n_services):
                row[idx_x_ij_alpha(i, j)] = 1
            row[idx_x_i_c_aux(i)] = M_cost
            constraints.append(row)
            b_ub.append(M_cost)

        # Convert constraints to numpy arrays
        A_ub = np.vstack(constraints)
        b_ub = np.array(b_ub)

        # Integer constraints for x_j^S and x_i^{c,aux} variables
        integrality = np.zeros(n_vars)
        for j in range(n_services):
            integrality[idx_x_j_S(j)] = 1  # x_j^S variables are binary
        for i in range(n_validators):
            integrality[idx_x_i_c_aux(i)] = 1  # x_i^{c,aux} variables are binary

        # Bounds for all variables
        lb = np.zeros(n_vars)  # lower bounds
        ub = np.ones(n_vars)  # upper bounds
        # x_j^S ∈ {0,1}
        for j in range(n_services):
            lb[idx_x_j_S(j)] = 0
            ub[idx_x_j_S(j)] = 1
        # x_ij^α ≥ 0
        for i in range(n_validators):
            for j in range(n_services):
                lb[idx_x_ij_alpha(i, j)] = 0
                ub[idx_x_ij_alpha(i, j)] = self.allocations[i, j]
        # x_i^c ≥ 0
        for i in range(n_validators):
            lb[idx_x_i_c(i)] = 0
            ub[idx_x_i_c(i)] = self.validator_stakes[i]
        # x_i^{c,aux} ∈ {0,1}
        for i in range(n_validators):
            lb[idx_x_i_c_aux(i)] = 0
            ub[idx_x_i_c_aux(i)] = 1

        # Solve the MILP
        result = milp(
            c=-c,  # Negative because scipy minimizes
            constraints=LinearConstraint(
                A_ub, 
                lb=-np.inf * np.ones(len(b_ub)),  # Lower bound is -inf for ≤ constraints
                ub=b_ub,  # Upper bound is b_ub
            ),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )

        if not result.success:
            return True, -result.fun, None, None

        # Extract results
        x = result.x
        profit = -result.fun
        attacked_services = np.array([x[idx_x_j_S(j)] > 0.5 for j in range(n_services)])
        attack_stakes = np.array([[x[idx_x_ij_alpha(i, j)] for j in range(n_services)] 
                                 for i in range(n_validators)])
        
        return result.fun <= 0, profit, attacked_services, attack_stakes

    def check_robustness_with_milp(self, loss_threshold: float = 0) -> tuple[float, npt.NDArray[np.bool_]]:
        """
        Calculate the maximum fraction of Byzantine services such that the network remains secure.
        
        Returns:
            Tuple of:
            - byzantine_fraction: The minimum fraction of Byzantine services needed to make the network insecure
            - byzantine_mask: Boolean array indicating which services are Byzantine in the optimal solution
        """
        n_validators = len(self.validator_stakes)
        n_services = len(self.service_attack_rewards)

        # Variable indices in the solution vector x:
        idx_z_B = lambda: 0  # z_B (byzantine services auxiliary)

        # First group: validator-related variables
        base_validator = 1
        idx_x_i_c = lambda i: base_validator + i  # x_i^c (validator cost)
        idx_x_i_c_aux = lambda i: base_validator + n_validators + i  # x_i^{c,aux} (validator cost auxiliary)
        idx_x_i_r = lambda i: base_validator + 2*n_validators + i  # x_i^r (remaining stake)
        idx_x_i_r_aux = lambda i: base_validator + 3*n_validators + i  # x_i^{r,aux} (remaining stake auxiliary)
        
        # Second group: service-related variables
        base_service = 4*n_validators + 1
        idx_x_j_S = lambda j: base_service + j  # x_j^S (attacked service)
        idx_x_j_B = lambda j: base_service + n_services + j  # x_j^B (byzantine service)
        
        # Third group: validator-service pair variables
        base_pair = base_service + 2*n_services + 1
        idx_x_ij_alpha = lambda i, j: base_pair + i*n_services + j  # x_ij^α (attack stake)
        idx_x_ij_r = lambda i, j: base_pair + n_validators*n_services + i*n_services + j  # x_ij^r (remaining allocation)
        idx_x_ij_r_aux = lambda i, j: base_pair + 2*n_validators*n_services + i*n_services + j  # x_ij^{r,aux} (remaining allocation auxiliary)

        n_vars = base_pair + 3*n_validators*n_services

        def print_solution(x):
            print("z_B", x[idx_z_B()])
            print("x_i^c", [x[idx_x_i_c(i)] for i in range(n_validators)])
            print("x_i^r", [x[idx_x_i_r(i)] for i in range(n_validators)])
            print("x_j^S", [x[idx_x_j_S(j)] for j in range(n_services)])
            print("x_j^B", [x[idx_x_j_B(j)] for j in range(n_services)])
            print("x_ij^α", [[x[idx_x_ij_alpha(i, j)] for j in range(n_services)] for i in range(n_validators)])
            print("x_ij^r", [[x[idx_x_ij_r(i, j)] for j in range(n_services)] for i in range(n_validators)])
            print("x_ij^r_aux", [[x[idx_x_ij_r_aux(i, j)] for j in range(n_services)] for i in range(n_validators)])

        # Objective: Minimize sum(R_j/θ_j * x_j^B)
        c = np.zeros(n_vars)
        for j in range(n_services):
            c[idx_x_j_B(j)] = self.service_attack_rewards[j] / self.service_attack_thresholds[j]
        
        # Normalize the objective function to be from 0 to 1
        c /= c.sum()

        # Initialize constraints
        constraints = []
        b_ub = []

        # Calculate M values
        M_feasibility = max(self.service_attack_thresholds[j] * np.sum(self.allocations[:, j]) 
                        for j in range(n_services))
        M_cost = max(np.max(self.validator_stakes), np.max(np.sum(self.allocations, axis=1)))
        M_remaining_stake = max(np.max(self.validator_stakes), np.max(np.sum(self.allocations, axis=1)))
        M_remaining_alloc = np.max(self.validator_stakes)
        M_byzantine_services = n_services

        # At least one service must be attacked if z_B = 0: sum(x_j^S) >= 1 - M_byzantine_services * z_B
        row = np.zeros(n_vars)
        for j in range(n_services):
            row[idx_x_j_S(j)] = -1
        row[idx_z_B()] = -M_byzantine_services
        constraints.append(row)
        b_ub.append(-1)

        # Or all services are be Byzantine if z_B = 1: sum(x_j^B) >= n_services - M_byzantine_services * (1 - z_B)
        row = np.zeros(n_vars)
        for j in range(n_services):
            row[idx_x_j_B(j)] = -1
        row[idx_z_B()] = M_byzantine_services
        constraints.append(row)
        b_ub.append(M_byzantine_services - n_services)

        # Attack must cost less than loss threshold: sum(R_j * x_j^S) - sum(x_i^c) >= -loss_threshold
        row = np.zeros(n_vars)
        for j in range(n_services):
            row[idx_x_j_S(j)] = -self.service_attack_rewards[j]
        for i in range(n_validators):
            row[idx_x_i_c(i)] = 1
        constraints.append(row)
        b_ub.append(loss_threshold)

        # Validator cost constraints
        for i in range(n_validators):
            # x_i^c <= x_i^r
            row = np.zeros(n_vars)
            row[idx_x_i_c(i)] = 1
            row[idx_x_i_r(i)] = -1
            constraints.append(row)
            b_ub.append(0)

            # x_i^c <= sum(x_ij^α)
            row = np.zeros(n_vars)
            row[idx_x_i_c(i)] = 1
            for j in range(n_services):
                row[idx_x_ij_alpha(i, j)] = -1
            constraints.append(row)
            b_ub.append(0)

            # x_i^c >= x_i^r - M_cost * z_i
            row = np.zeros(n_vars)
            row[idx_x_i_c(i)] = -1
            row[idx_x_i_r(i)] = 1
            row[idx_x_i_c_aux(i)] = -M_cost
            constraints.append(row)
            b_ub.append(0)

            # x_i^c >= sum(x_ij^α) - M_cost * (1-z_i)
            row = np.zeros(n_vars)
            row[idx_x_i_c(i)] = -1
            for j in range(n_services):
                row[idx_x_ij_alpha(i, j)] = 1
            row[idx_x_i_c_aux(i)] = M_cost
            constraints.append(row)
            b_ub.append(M_cost)

        # Remaining stake constraints
        for i in range(n_validators):
            # x_i^r >= stake_i - sum(w_ij * x_j^B)
            row = np.zeros(n_vars)
            row[idx_x_i_r(i)] = -1
            for j in range(n_services):
                row[idx_x_j_B(j)] = -self.allocations[i, j]
            constraints.append(row)
            b_ub.append(-self.validator_stakes[i])

            # x_i^r <= stake_i - sum(w_ij * x_j^B) + M * y_i
            row = np.zeros(n_vars)
            row[idx_x_i_r(i)] = 1
            for j in range(n_services):
                row[idx_x_j_B(j)] = self.allocations[i, j]
            row[idx_x_i_r_aux(i)] = -M_remaining_stake
            constraints.append(row)
            b_ub.append(self.validator_stakes[i])

            # x_i^r <= M * (1-y_i)
            row = np.zeros(n_vars)
            row[idx_x_i_r(i)] = 1
            row[idx_x_i_r_aux(i)] = M_remaining_stake
            constraints.append(row)
            b_ub.append(M_remaining_stake)

        # Service attack constraints
        for j in range(n_services):
            # x_j^S + x_j^B <= 1
            row = np.zeros(n_vars)
            row[idx_x_j_S(j)] = 1
            row[idx_x_j_B(j)] = 1
            constraints.append(row)
            b_ub.append(1)

            # Attack feasibility: sum(x_ij^α) >= θ_j * sum(x_ij^r) - M * (1-x_j^S)
            row = np.zeros(n_vars)
            for i in range(n_validators):
                row[idx_x_ij_alpha(i, j)] = -1
                row[idx_x_ij_r(i, j)] = self.service_attack_thresholds[j]
            row[idx_x_j_S(j)] = M_feasibility
            constraints.append(row)
            b_ub.append(M_feasibility)

        for i in range(n_validators):
            for j in range(n_services):
                # Remaining allocation constraints
                # x_ij^r <= w_ij
                row = np.zeros(n_vars)
                row[idx_x_ij_r(i, j)] = 1
                constraints.append(row)
                b_ub.append(self.allocations[i, j])

                # x_ij^r <= x_i^r
                row = np.zeros(n_vars)
                row[idx_x_ij_r(i, j)] = 1
                row[idx_x_i_r(i)] = -1
                constraints.append(row)
                b_ub.append(0)

                # x_ij^r >= w_ij - M * z_ij
                row = np.zeros(n_vars)
                row[idx_x_ij_r(i, j)] = -1
                row[idx_x_ij_r_aux(i, j)] = -M_remaining_alloc
                constraints.append(row)
                b_ub.append(-self.allocations[i, j])

                # x_ij^r >= x_i^r - M * (1-z_ij)
                row = np.zeros(n_vars)
                row[idx_x_ij_r(i, j)] = -1
                row[idx_x_i_r(i)] = 1
                row[idx_x_ij_r_aux(i, j)] = M_remaining_alloc
                constraints.append(row)
                b_ub.append(M_remaining_alloc)

                # Attack stake constraints
                # x_ij^α <= x_ij^r
                row = np.zeros(n_vars)
                row[idx_x_ij_alpha(i, j)] = 1
                row[idx_x_ij_r(i, j)] = -1
                constraints.append(row)
                b_ub.append(0)

        # Convert constraints to numpy arrays
        A_ub = np.vstack(constraints)
        b_ub = np.array(b_ub)

        # Integer constraints
        integrality = np.zeros(n_vars)
        for i in range(n_validators):
            integrality[idx_x_i_c_aux(i)] = 1
            integrality[idx_x_i_r_aux(i)] = 1
        for j in range(n_services):
            integrality[idx_x_j_S(j)] = 1
            integrality[idx_x_j_B(j)] = 1
        integrality[idx_z_B()] = 1
        for i in range(n_validators):
            for j in range(n_services):
                integrality[idx_x_ij_r_aux(i, j)] = 1

        # Variable bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        # Update upper bounds for continuous variables
        for i in range(n_validators):
            ub[idx_x_i_c(i)] = self.validator_stakes[i]
            ub[idx_x_i_r(i)] = self.validator_stakes[i]
        for i in range(n_validators):
            for j in range(n_services):
                ub[idx_x_ij_alpha(i, j)] = self.allocations[i, j]
                ub[idx_x_ij_r(i, j)] = self.allocations[i, j]

        # Solve the MILP
        # Try with presolve first and if that fails, try without presolve
        for presolve in [True, False]:
            result = milp(
                c=c,
                constraints=LinearConstraint(
                    A_ub,
                    lb=-np.inf * np.ones(len(b_ub)),
                    ub=b_ub,
                ),
                integrality=integrality,
                bounds=Bounds(lb, ub),
                options=dict(presolve=presolve)
            )
            
            if result.success:
                break

        if not result.success:
            print(self.__dict__)
            print(loss_threshold)
            print(result.message)
            raise RuntimeError("Failed to solve MILP")

        # Extract results
        x = result.x
        byzantine_services = np.array([x[idx_x_j_B(j)] > 0.5 for j in range(n_services)])
        byzantine_fraction = result.fun

        return byzantine_fraction, byzantine_services
        



class FullySymmetricRestakingNetwork:
    def __init__(self, n_validators: int, n_services: int, validator_stake: float, allocation_per_service: float, service_attack_reward: float, service_attack_threshold: float, has_base_service: bool = False, base_service_reward: Optional[float] = None):
        self.n_validators = n_validators
        self.n_services = n_services
        self.validator_stake = validator_stake
        self.allocation_per_service = allocation_per_service
        self.service_attack_reward = service_attack_reward
        self.service_attack_threshold = service_attack_threshold
        self.has_base_service = has_base_service
        self.base_service_reward = base_service_reward

        assert not (self.has_base_service and self.base_service_reward is None), "Base service reward must be provided if base service is present"

    def check_security(self) -> tuple[bool, float]:
        num_consolidated_attacking_validators = math.floor(self.n_validators * self.service_attack_threshold)
        partial_validator_fraction = self.n_validators * self.service_attack_threshold - num_consolidated_attacking_validators
        require_partial_validator = partial_validator_fraction > 0
        
        if self.has_base_service:
            attack_base_service_options = [False, True]
        else:
            attack_base_service_options = [False]

        potential_profit = []
        for attack_base_service in attack_base_service_options:
            for num_attacked_services in range(0, self.n_services + 1):
                if num_attacked_services == 0 and not attack_base_service:
                    continue

                consolidated_validator_allocations = self.allocation_per_service * num_attacked_services
                partial_validator_allocations = partial_validator_fraction * self.allocation_per_service * num_attacked_services

                if attack_base_service:
                    consolidated_validator_allocations += self.validator_stake
                    partial_validator_allocations += partial_validator_fraction * self.validator_stake

                cost = num_consolidated_attacking_validators * min(self.validator_stake, consolidated_validator_allocations)
                if require_partial_validator:
                        cost += min(partial_validator_allocations, self.validator_stake)

                prizes = self.service_attack_reward * num_attacked_services
                if attack_base_service:
                    prizes += self.base_service_reward

                attack_profit = prizes - cost
                potential_profit.append(attack_profit)

        best_profit = max(potential_profit)
        return best_profit < 0, best_profit
    
    def apply_byzantine_services(self, n_byzantine_services: int) -> "FullySymmetricRestakingNetwork":
        remaining_validator_stake = max(0, self.validator_stake - n_byzantine_services * self.allocation_per_service)
        remaining_allocation_per_service = min(remaining_validator_stake, self.allocation_per_service)
        return FullySymmetricRestakingNetwork(
            n_validators=self.n_validators,
            n_services=self.n_services - n_byzantine_services,
            validator_stake=remaining_validator_stake,
            allocation_per_service=remaining_allocation_per_service,
            service_attack_reward=self.service_attack_reward,
            service_attack_threshold=self.service_attack_threshold,
            has_base_service=self.has_base_service,
            base_service_reward=self.base_service_reward
        )


if __name__ == "__main__":
    service_attack_rewards = np.ones(3)
    service_attack_thresholds = 0.5 * np.ones(3)
    validator_stakes = 2 * np.ones(3)
    allocations = 2 * np.ones((3, 3))

    restaking_network = RestakingNetwork(service_attack_rewards, service_attack_thresholds, validator_stakes, allocations)
    robustness, byzantine_mask = restaking_network.check_robustness_with_milp()
    print(robustness, byzantine_mask)

    restaking_network = restaking_network.apply_byzantine_services(byzantine_mask)
    print(restaking_network.check_security_with_milp())
