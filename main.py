from concurrent.futures import ProcessPoolExecutor
import dataclasses
import math
from typing import List, Literal, Optional, Tuple, Union
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from restaking_network import RestakingNetwork, FullySymmetricRestakingNetwork


@dataclasses.dataclass
class RestakingNetworkInfo:
    network_type: Literal['elastic', 'rigid'] = 'elastic'
    num_validators: Optional[int] = None
    num_services: Optional[int] = None
    service_attack_reward: Optional[float] = None
    service_attack_threshold: Optional[float] = None
    restaking_degree: Optional[float] = None
    validator_stake: Optional[float] = None
    base_service: Optional[Tuple[float, float]] = None


def create_restaking_network(info: RestakingNetworkInfo, symmetric_network: bool) -> Union[RestakingNetwork, FullySymmetricRestakingNetwork]:
    if symmetric_network:
        if info.network_type != 'elastic':
            raise ValueError("Fully symmetric network is only supported for elastic networks")
        
        restaking_degree = info.restaking_degree
        if info.base_service is not None:
            assert info.base_service[1] == info.service_attack_threshold, "Base service threshold must be equal to service attack threshold in a fully symmetric network"
            restaking_degree -= 1
            base_service_reward = info.base_service[0]
        else:
            base_service_reward = None

        allocation_per_service = restaking_degree / info.num_services * info.validator_stake

        return FullySymmetricRestakingNetwork(
            n_validators=info.num_validators,
            n_services=info.num_services,
            validator_stake=info.validator_stake,
            allocation_per_service=allocation_per_service,
            service_attack_reward=info.service_attack_reward,
            service_attack_threshold=info.service_attack_threshold,
            has_base_service=info.base_service is not None,
            base_service_reward=base_service_reward
        )

    if info.restaking_degree is None:
        raise ValueError("Restaking degree must be provided")

    service_attack_rewards = info.service_attack_reward * np.ones(info.num_services)
    service_attack_thresholds = info.service_attack_threshold * np.ones(info.num_services)
    
    if info.base_service is not None:
        info.restaking_degree -= 1
    
    if info.network_type == 'elastic':
        validator_stakes = info.validator_stake * np.ones(info.num_validators)
        allocations = info.restaking_degree / info.num_services * info.validator_stake * np.ones((info.num_validators, info.num_services))
    elif info.network_type == 'rigid':
        if info.restaking_degree.is_integer():
            validator_stakes, allocations = create_rigid_portions(info.num_validators, info.num_services, info.validator_stake, int(info.restaking_degree))
        else:
            higher_restaking_degree = math.ceil(info.restaking_degree)
            lower_restaking_degree = math.floor(info.restaking_degree)
            lower_validator_stake = info.validator_stake * (higher_restaking_degree - info.restaking_degree)
            higher_validator_stake = info.validator_stake * (info.restaking_degree - lower_restaking_degree)
            
            lower_validator_stakes, lower_allocations = create_rigid_portions(info.num_validators, info.num_services, lower_validator_stake, lower_restaking_degree)
            higher_validator_stakes, higher_allocations = create_rigid_portions(info.num_validators, info.num_services, higher_validator_stake, higher_restaking_degree)

            validator_stakes = np.concatenate([lower_validator_stakes, higher_validator_stakes])
            allocations = np.concatenate([lower_allocations, higher_allocations], axis=0)
    else:
        raise ValueError(f"Invalid network type: {info.network_type}")
    
    
    if info.base_service is not None:
        info.restaking_degree += 1
        service_attack_rewards = np.concatenate([
            np.array([info.base_service[0]]),
            service_attack_rewards
        ])
        service_attack_thresholds = np.concatenate([
            np.array([info.base_service[1]]),
            service_attack_thresholds
        ])
        allocations = np.concatenate([
            validator_stakes.reshape(-1, 1),
            allocations
        ], axis=1)

    return RestakingNetwork(service_attack_rewards, service_attack_thresholds, validator_stakes, allocations)


def create_rigid_portions(num_validators: int, num_services: int, validator_stake: float, restaking_degree: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    number_of_portions = math.comb(num_services, restaking_degree)

    stake_per_portion = validator_stake / number_of_portions

    validator_stakes = stake_per_portion * np.ones(num_validators * number_of_portions)
    allocations = stake_per_portion * np.tile(np.array(list(
        multiset_permutations([1] * restaking_degree + [0] * (num_services - restaking_degree))
    )), (num_validators, 1))

    return validator_stakes, allocations


def get_start_max_stake(info: RestakingNetworkInfo, multiplier: float = 2) -> float:
    return multiplier * info.service_attack_reward * info.num_services / (info.service_attack_threshold * info.num_validators)


def search_for_minimum_stake(info: RestakingNetworkInfo, robustness_threshold: float, loss_threshold: float = 0, max_stake_multiplier: float = 2, symmetric_network: bool = False, verbose: bool = False) -> Optional[float]:
    start_min_stake = 0
    start_max_stake = get_start_max_stake(info, max_stake_multiplier)

    if verbose:
        print("Starting search for minimum stake")
        print(info)
        print(f"robustness_threshold: {robustness_threshold}")
        print(f"start_min_stake: {start_min_stake}, start_max_stake: {start_max_stake}")

    min_stake = start_min_stake
    max_stake = start_max_stake
    while max_stake - min_stake > 1e-6:
        mid_stake = (min_stake + max_stake) / 2

        if verbose:
            print(f"Checking stake: {mid_stake}")

        info = dataclasses.replace(info, validator_stake=mid_stake)
        restaking_network = create_restaking_network(info, symmetric_network)

        if isinstance(restaking_network, FullySymmetricRestakingNetwork):
            if robustness_threshold > 0:
                restaking_network = restaking_network.apply_byzantine_services(int(robustness_threshold * restaking_network.n_services + 1e-6))

            _, profit = restaking_network.check_security()
            add_stake = profit > -loss_threshold

            if verbose:
                print(f"profit: {profit}")
                print(f"add_stake: {add_stake}")
        elif robustness_threshold > 0:
            robustness, _ = restaking_network.check_robustness_with_milp(loss_threshold=loss_threshold)
            add_stake = robustness <= robustness_threshold + 1e-9

            if verbose:
                print(f"robustness: {robustness}")
                print(f"add_stake: {add_stake}")
        else:
            _, profit, attacked_services, attack_stake = restaking_network.check_security_with_milp()
            add_stake = profit > -loss_threshold

            if verbose:
                print(f"profit: {profit}")
                print(f"add_stake: {add_stake}")

        if add_stake:
            min_stake = mid_stake
        else:
            max_stake = mid_stake

    result = (min_stake + max_stake) / 2

    if result > start_max_stake - 1e-6:
        return None
    else:
        return result
    
    
def search_for_minimum_loss(info: RestakingNetworkInfo, robustness_threshold: float, max_loss_threshold: float = 0,\
                            symmetric_network: bool = False, verbose: bool = False) -> Optional[float]:
    start_min_loss = 0
    start_max_loss = max_loss_threshold

    if verbose:
        print("Starting search for minimum loss")
        print(info)
        print(f"robustness_threshold: {robustness_threshold}")
        print(f"start_min_loss: {start_min_loss}, start_max_loss: {start_max_loss}")

    min_loss = start_min_loss
    max_loss = start_max_loss
    while max_loss - min_loss > 1e-6:
        mid_loss = (min_loss + max_loss) / 2

        if verbose:
            print(f"Checking loss: {mid_loss}")

        restaking_network = create_restaking_network(info, symmetric_network)

        
        if isinstance(restaking_network, FullySymmetricRestakingNetwork):
            if robustness_threshold > 0:
                restaking_network = restaking_network.apply_byzantine_services(int(robustness_threshold * restaking_network.n_services + 1e-6))

            _, profit = restaking_network.check_security()
            add_loss = profit <= -mid_loss

            if verbose:
                print(f"profit: {profit}")
                print(f"add_loss: {add_loss}")
        elif robustness_threshold > 0:
            robustness, _ = restaking_network.check_robustness_with_milp(loss_threshold=mid_loss)
            add_loss = robustness > robustness_threshold + 1e-9

            if verbose:
                print(f"robustness: {robustness}")
                print(f"add_loss: {add_loss}")
        else:
            _, profit, _, _ = restaking_network.check_security_with_milp()
            add_loss = profit <= -mid_loss

            if verbose:
                print(f"profit: {profit}")
                print(f"add_loss: {add_loss}")


        # if robustness_threshold > 0 and not symmetric_network:
        #     robustness, _ = restaking_network.check_robustness_with_milp(loss_threshold=mid_loss)
        #     add_loss = robustness > robustness_threshold + 1e-9

        #     if verbose:
        #         print(f"robustness: {robustness}")
        #         print(f"add_loss: {add_loss}")
        # else:
        #     if robustness_threshold > 0 and symmetric_network:
        #         num_services = len(restaking_network.service_names)
        #         num_byzantine_services = int(robustness_threshold * num_services + 1e-6)
        #         byzantine_services = np.concatenate([
        #             np.zeros(num_services - num_byzantine_services),
        #             np.ones(num_byzantine_services)
        #         ])
        #         restaking_network = restaking_network.apply_byzantine_services(byzantine_services)

        #     _, profit, _, _ = restaking_network.check_security_with_milp()
        #     add_loss = profit <= -mid_loss

        #     if verbose:
        #         print(f"profit: {profit}")
        #         print(f"add_loss: {add_loss}")

        if add_loss:
            min_loss = mid_loss
        else:
            max_loss = mid_loss

    result = (min_loss + max_loss) / 2

    if result > start_max_loss - 1e-6:
        return None
    else:
        return result


def search_for_minimum_stake_wrapper(info: RestakingNetworkInfo, restaking_degree: float, robustness_threshold: float, loss_threshold: float = 0, max_stake_multiplier: float = 2, symmetric_network: bool = False) -> tuple[float, float, float]:
    info = dataclasses.replace(info, restaking_degree=restaking_degree)
    return restaking_degree, robustness_threshold, search_for_minimum_stake(info, robustness_threshold, loss_threshold, max_stake_multiplier, symmetric_network=symmetric_network)


def build_graph_y_axis_stake(filename: str, info: RestakingNetworkInfo, restaking_degrees: np.ndarray,
                             robustness_thresholds: np.ndarray, loss_threshold: float = 0, max_stake_multiplier: float = 2,
                             symmetric_network: bool = False, results_folder: str = 'results') -> None:
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(search_for_minimum_stake_wrapper, info, restaking_degree, robustness_threshold, loss_threshold, max_stake_multiplier=max_stake_multiplier, symmetric_network=symmetric_network)
            for robustness_threshold in robustness_thresholds
            for restaking_degree in restaking_degrees
        ]
        
        for future in futures:
            results.append(future.result())

    df = pd.DataFrame(results, columns=["restaking_degree", "robustness_threshold", "min_stake"])
    print(df)
    df.to_csv(f"{results_folder}/{filename}.csv", index=False)
    
    plt.figure()
    
    ax = sns.lineplot(x="restaking_degree", y="min_stake", data=df, hue="robustness_threshold", palette="deep", alpha=0.8)
    ax.set_xticklabels([f'{float(x.get_text()):.2f}' for x in ax.get_xticklabels()])
    
    legend = ax.legend(title="Robustness Threshold")
    for text in legend.get_texts():
        text.set_text(f'{float(text.get_text()):.2f}')

    ax.set_xlabel("Restaking Degree")
    ax.set_ylabel("Minimum Stake")

    ax.axhline(y=get_start_max_stake(info, multiplier=max_stake_multiplier), color='k', linestyle='--', label="Max Stake")

    ax.grid(color='gray', linewidth=0.5, alpha=0.5)

    plt.savefig(f"{results_folder}/{filename}.png", dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()

def search_for_minimum_loss_wrapper(info: RestakingNetworkInfo, robustness_threshold: float, validator_stake: float, max_loss_threshold: float, symmetric_network: bool = False) -> tuple[float, float, float]:
    info = dataclasses.replace(info, validator_stake=validator_stake)
    return validator_stake, robustness_threshold, search_for_minimum_loss(info, robustness_threshold, max_loss_threshold, symmetric_network=symmetric_network)

def build_graph_y_axis_loss(filename: str, info: RestakingNetworkInfo, robustness_thresholds: np.ndarray,
                            validator_stakes: np.ndarray, max_loss_threshold: float, symmetric_network: bool = False,
                            results_folder: str = 'results') -> None:
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(search_for_minimum_loss_wrapper, info, robustness_threshold, validator_stake, max_loss_threshold, symmetric_network=symmetric_network)
            for robustness_threshold in robustness_thresholds
            for validator_stake in validator_stakes
        ]

        for future in futures:
            results.append(future.result())

    df = pd.DataFrame(results, columns=["validator_stake", "robustness_threshold", "loss_threshold"])
    print(df)
    df.to_csv(f"{results_folder}/{filename}.csv", index=False)
    
    plt.figure()
    
    ax = sns.lineplot(x="validator_stake", y="loss_threshold", data=df, hue="robustness_threshold", palette="deep", alpha=0.8)
    ax.set_xticklabels([f'{float(x.get_text()):.2f}' for x in ax.get_xticklabels()])
    
    legend = ax.legend(title="Robustness Threshold")
    for text in legend.get_texts():
        text.set_text(f'{float(text.get_text()):.2f}')

    ax.set_xlabel("Validator Stake")
    ax.set_ylabel("Min Loss")

    ax.axhline(y=max_loss_threshold, color='k', linestyle='--', label="Max Loss")

    ax.grid(color='gray', linewidth=0.5, alpha=0.5)

    plt.savefig(f"{results_folder}/{filename}.png", dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


######################################
# Figures
######################################


results_folder_for_figures = 'figures'


######################################
# Figure 3
######################################


def figure3_wrapper(info: RestakingNetworkInfo, service_attack_threshold: float, restaking_degree: float) -> tuple[float, float, float]:
    info = dataclasses.replace(info, service_attack_threshold=service_attack_threshold)
    info = dataclasses.replace(info, restaking_degree=restaking_degree)
    return service_attack_threshold, restaking_degree, search_for_minimum_stake(info, robustness_threshold=0, loss_threshold=0, max_stake_multiplier=2, symmetric_network=True)


def build_figure3_graph(filename: str, info: RestakingNetworkInfo) -> None:
    restaking_degrees = [float(rd) for rd in np.arange(1, info.num_services, 0.01)]
    service_attack_thresholds = [1/3, 1/2]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(figure3_wrapper, info, service_attack_threshold, restaking_degree)
            for service_attack_threshold in service_attack_thresholds
            for restaking_degree in restaking_degrees
        ]
        
        for future in futures:
            results.append(future.result())

    df = pd.DataFrame(results, columns=["service_attack_threshold", "restaking_degree", "min_stake"])
    csv_df = df.pivot(
        index='restaking_degree',
        columns='service_attack_threshold',
        values='min_stake'
    ).reset_index()
    
    csv_df.columns.name = None
    csv_df = csv_df.rename(columns={
        threshold: f"min_stake_threshold_{threshold:.2f}"
        for threshold in service_attack_thresholds
    })
    csv_df.to_csv(f"{results_folder_for_figures}/{filename}.csv", index=False)

    plt.figure()
    
    ax = sns.lineplot(x="restaking_degree", y="min_stake", data=df, hue="service_attack_threshold", palette="deep", alpha=0.8)
    ax.set_xticklabels([f'{float(x.get_text()):.2f}' for x in ax.get_xticklabels()])
    
    legend = ax.legend(title="Service Attack Threshold")
    for text in legend.get_texts():
        text.set_text(f'{float(text.get_text()):.2f}')

    ax.set_xlabel("Restaking Degree")
    ax.set_ylabel("Minimum Stake")

    ax.axhline(y=get_start_max_stake(dataclasses.replace(info, service_attack_threshold=1/2), multiplier=2), color='k', linestyle='--', label="Max Stake")

    ax.grid(color='gray', linewidth=0.5, alpha=0.5)

    plt.savefig(f"{results_folder_for_figures}/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()


def figure3() -> None:
    for n in [10, 11, 12]:
        info = RestakingNetworkInfo(
            num_validators=n,
            num_services=n,
            service_attack_reward=1,
        )
        build_figure3_graph(f"figure3_n{n}", info)
    

######################################
# Figure 4
######################################


def figure4_stake_wrapper(info: RestakingNetworkInfo, restaking_degree: float, robustness_threshold: float, budget: float) -> tuple[float, float, float]:
    info = dataclasses.replace(info, restaking_degree=restaking_degree)
    return restaking_degree, robustness_threshold, search_for_minimum_stake(info, robustness_threshold=robustness_threshold, loss_threshold=budget, max_stake_multiplier=3.33, symmetric_network=True)


def build_figure4_stake_graph(filename: str, info: RestakingNetworkInfo, budget: float) -> None:
    restaking_degrees = [float(rd) for rd in np.arange(1, info.num_services, 0.01)]
    robustness_thresholds = [float(rt) for rt in np.linspace(0, 1 - 1/info.num_services, info.num_services)]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(figure4_stake_wrapper, info, restaking_degree, robustness_threshold, budget)
            for robustness_threshold in robustness_thresholds
            for restaking_degree in restaking_degrees
        ]
        
        for future in futures:
            results.append(future.result())

    df = pd.DataFrame(results, columns=["restaking_degree", "robustness_threshold", "min_stake"])
    csv_df = df.pivot(
        index='restaking_degree',
        columns='robustness_threshold',
        values='min_stake'
    ).reset_index()
    
    csv_df.columns.name = None
    csv_df = csv_df.rename(columns={
        threshold: f"min_stake_threshold_{threshold:.2f}"
        for threshold in robustness_thresholds
    })
    csv_df.to_csv(f"{results_folder_for_figures}/{filename}.csv", index=False)

    plt.figure()
    
    ax = sns.lineplot(x="restaking_degree", y="min_stake", data=df, hue="robustness_threshold", palette="deep", alpha=0.8)
    ax.set_xticklabels([str(float(x.get_text().replace('\u2212', '-'))).format('.2f') for x in ax.get_xticklabels()])
    
    legend = ax.legend(title="Robustness Threshold")
    for text in legend.get_texts():
        text.set_text(f'{float(text.get_text()):.2f}')

    ax.set_xlabel("Restaking Degree")
    ax.set_ylabel("Minimum Stake")

    ax.axhline(y=get_start_max_stake(info, multiplier=3.33), color='k', linestyle='--', label="Max Stake")

    ax.grid(color='gray', linewidth=0.5, alpha=0.5)

    plt.savefig(f"{results_folder_for_figures}/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()


def figure4() -> None:
    n=15
    for budget in [0, 1, 2]:
        for base_service in [None, (10, 1/3)]:
            info = RestakingNetworkInfo(
                num_validators=n,
                num_services=n,
                service_attack_reward=1,
                service_attack_threshold=1/3,
                base_service=base_service
            )
            build_figure4_stake_graph(f"figure4_y_stake_budget_{budget}{'' if base_service is None else f'_base_service_{base_service[0]}_{base_service[1]:.2f}'}", info, budget)


######################################
# Figure 5
######################################


def figure5_budget_wrapper(info: RestakingNetworkInfo, restaking_degree: float, robustness_threshold: float) -> tuple[float, float, float]:
    info = dataclasses.replace(info, restaking_degree=restaking_degree)
    return restaking_degree, robustness_threshold, search_for_minimum_loss(info, robustness_threshold=robustness_threshold, max_loss_threshold=info.validator_stake * info.num_validators / 2, symmetric_network=True)


def build_figure5_graph(filename: str, info: RestakingNetworkInfo, restaking_degrees: List[float]) -> None:
    n = info.num_services
    robustness_thresholds = [float(rt) for rt in np.linspace(0, 1 - 1/n, n)]
    
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(figure5_budget_wrapper, info, restaking_degree, robustness_threshold)
            for robustness_threshold in robustness_thresholds
            for restaking_degree in restaking_degrees
        ]
        
        for future in futures:
            results.append(future.result())

    df = pd.DataFrame(results, columns=["restaking_degree", "robustness_threshold", "min_budget"])
    csv_df = df.pivot(
        index='robustness_threshold',
        columns='restaking_degree',
        values='min_budget'
    ).reset_index()
    
    csv_df.columns.name = None
    csv_df = csv_df.rename(columns={
        rd: f"min_budget_{rd:.2f}"
        for rd in restaking_degrees
    })
    csv_df.to_csv(f"{results_folder_for_figures}/{filename}.csv", index=False)

    plt.figure()
    
    ax = sns.lineplot(x="robustness_threshold", y="min_budget", data=df, hue="restaking_degree", palette="deep", alpha=0.8)
    
    ax.legend(title="Restaking Degree")

    ax.set_xlabel("Byzantine Services")
    ax.set_ylabel("Minimum Budget")

    ax.grid(color='gray', linewidth=0.5, alpha=0.5)

    plt.savefig(f"{results_folder_for_figures}/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()


def figure5() -> None:
    info = RestakingNetworkInfo(
        num_validators=15,
        num_services=15,
        validator_stake=10,
        service_attack_reward=1,
        service_attack_threshold=1/3,
    )
    restaking_degrees = [float(rd) for rd in np.arange(1, 3.25, 0.25)]
    build_figure5_graph(f"figure5", info, restaking_degrees)


######################################
# Figure 6
######################################


def figure6_budget_wrapper(info: RestakingNetworkInfo, restaking_degree: float, robustness_threshold: float, name: str) -> tuple[float, float, str, float]:
    info = dataclasses.replace(info, restaking_degree=restaking_degree)
    return restaking_degree, robustness_threshold, name, search_for_minimum_loss(info, robustness_threshold=robustness_threshold, max_loss_threshold=info.validator_stake * info.num_validators / 2, symmetric_network=True)


def build_figure6_graph(filename: str, infos: List[Tuple[str, RestakingNetworkInfo]]) -> None:
    n = infos[2][1].num_services
    restaking_degrees = [float(rd) for rd in np.arange(1, n, 0.01)]
    robustness_thresholds = [float(rt) for rt in np.linspace(0, 1 - 1/n, n)]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(figure6_budget_wrapper, network_info, restaking_degree, robustness_threshold, name)
            for robustness_threshold in robustness_thresholds
            for restaking_degree in restaking_degrees
            for (name, network_info) in infos
        ]
        
        for future in futures:
            results.append(future.result())

    df = pd.DataFrame(results, columns=["restaking_degree", "robustness_threshold", "type", "min_budget"])
    df = df.loc[df.groupby(["type", "robustness_threshold"])["min_budget"].idxmax()]
    csv_df = df.pivot(
        index='robustness_threshold',
        columns='type',
        values='min_budget'
    ).reset_index()
    
    csv_df.columns.name = None
    csv_df = csv_df.rename(columns={
        t: f"min_budget_{t}"
        for t in ["base_only", "no_base", "total"]
    })
    csv_df.to_csv(f"{results_folder_for_figures}/{filename}.csv", index=False)

    plt.figure()
    
    ax = sns.lineplot(x="robustness_threshold", y="min_budget", data=df, hue="type", palette="deep", alpha=0.8)
    
    ax.legend(title="Type")

    ax.set_xlabel("Byzantine Services")
    ax.set_ylabel("Minimum Budget")

    ax.grid(color='gray', linewidth=0.5, alpha=0.5)

    plt.savefig(f"{results_folder_for_figures}/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()


def figure6() -> None:
    info = RestakingNetworkInfo(
        num_validators=15,
        num_services=15,
        validator_stake=7.8,
        service_attack_reward=1,
        service_attack_threshold=1/3,
        base_service=(10, 1/3)
    )
    base_only_info = dataclasses.replace(
        info,
        base_service=None,
        num_services=1,
        service_attack_reward=info.base_service[0],
        service_attack_threshold=info.base_service[1],
        validator_stake=2.4
    )
    no_base_info = dataclasses.replace(
        info,
        base_service=None,
        validator_stake=5.4
    )
    build_figure6_graph(f"figure6", [("base_only", base_only_info), ("no_base", no_base_info), ("total", info)])


######################################
# Figure 7
######################################


def figure7_stake_wrapper(info: RestakingNetworkInfo, restaking_degree: float, robustness_threshold: float, budget: float, use_milp: bool) -> tuple[float, float, float]:
    info = dataclasses.replace(info, restaking_degree=restaking_degree)
    return restaking_degree, robustness_threshold, use_milp, search_for_minimum_stake(info, robustness_threshold=robustness_threshold, loss_threshold=budget, max_stake_multiplier=4, symmetric_network=not use_milp)


def build_figure7_stake_graph(filename: str, info: RestakingNetworkInfo, budget: float) -> None:
    restaking_degrees = [float(rd) for rd in np.arange(1, info.num_services, 0.01)]
    robustness_thresholds = [float(rt) for rt in np.linspace(0, 1 - 1/info.num_services, info.num_services)]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(figure7_stake_wrapper, info, restaking_degree, robustness_threshold, budget, use_milp)
            for use_milp in [True, False]
            for robustness_threshold in robustness_thresholds
            for restaking_degree in restaking_degrees
        ]
        
        for future in futures:
            results.append(future.result())

    df = pd.DataFrame(results, columns=["restaking_degree", "robustness_threshold", "use_milp", "min_stake"])
    csv_df = df.pivot(
        index='restaking_degree',
        columns=['use_milp', 'robustness_threshold'],
        values='min_stake'
    ).reset_index()
    
    csv_df.columns = [
        'restaking_degree' if col[0] == 'restaking_degree'
        else f"min_stake_threshold_{col[1]}_{'with' if col[0] else 'without'}_milp"
        for col in csv_df.columns
    ]

    csv_df.columns.name = None
    csv_df = csv_df.rename(columns={
        (use_milp, threshold): f"min_stake_threshold_{threshold:.2f}_{'with' if use_milp else 'without'}_milp"
        for use_milp in [True, False]
        for threshold in robustness_thresholds
    })
    csv_df.to_csv(f"{results_folder_for_figures}/{filename}.csv", index=False)

    plt.figure()
    
    ax = sns.lineplot(x="restaking_degree", y="min_stake", data=df, hue=df[['use_milp', 'robustness_threshold']].apply(tuple, axis=1), palette="deep", alpha=0.8)
    ax.set_xticklabels([f'{float(x.get_text()):.2f}' for x in ax.get_xticklabels()])
    
    legend = ax.legend(title="Robustness Threshold")

    ax.set_xlabel("Restaking Degree")
    ax.set_ylabel("Minimum Stake")

    ax.axhline(y=get_start_max_stake(info, multiplier=4), color='k', linestyle='--', label="Max Stake")

    ax.grid(color='gray', linewidth=0.5, alpha=0.5)

    plt.savefig(f"{results_folder_for_figures}/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()


def figure7() -> None:
    n=3
    for budget in [0, 1, 2]:
        info = RestakingNetworkInfo(
            num_validators=n,
            num_services=n,
            service_attack_reward=1,
            service_attack_threshold=1/3
        )
        build_figure7_stake_graph(f"figure7_budget_{budget}", info, budget)


######################################
# Figure 8
######################################


def figure8_stake_wrapper(info: RestakingNetworkInfo, restaking_degree: float, robustness_threshold: float, budget: float) -> tuple[float, float, float]:
    info = dataclasses.replace(info, restaking_degree=restaking_degree)
    return restaking_degree, robustness_threshold, search_for_minimum_stake(info, robustness_threshold=robustness_threshold, loss_threshold=budget, max_stake_multiplier=10, symmetric_network=info.base_service[1] == info.service_attack_threshold)


def build_figure8_stake_graph(filename: str, info: RestakingNetworkInfo, loss_threshold: float) -> None:
    restaking_degrees = [float(rd) for rd in np.arange(1, info.num_services, 0.01)]

    service_weights = info.service_attack_reward / info.service_attack_threshold * (1 + np.arange(0, info.num_services, 1))
    base_service_weight = info.base_service[0] / info.base_service[1]
    max_weight = service_weights[-1] + base_service_weight
    all_weights = np.concatenate([np.zeros(1), service_weights, service_weights + base_service_weight])
    robustness_thresholds = [float(rt) for rt in all_weights[:-1] / max_weight]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(figure8_stake_wrapper, info, restaking_degree, robustness_threshold, loss_threshold)
            for robustness_threshold in robustness_thresholds
            for restaking_degree in restaking_degrees
        ]
        
        for future in futures:
            results.append(future.result())

    df = pd.DataFrame(results, columns=["restaking_degree", "robustness_threshold", "min_stake"])
    csv_df = df.pivot(
        index='restaking_degree',
        columns='robustness_threshold',
        values='min_stake'
    ).reset_index()
    
    csv_df.columns.name = None
    csv_df = csv_df.rename(columns={
        threshold: f"min_stake_threshold_{threshold:.2f}"
        for threshold in robustness_thresholds
    })
    csv_df.to_csv(f"{results_folder_for_figures}/{filename}.csv", index=False)

    plt.figure()
    
    ax = sns.lineplot(x="restaking_degree", y="min_stake", data=df, hue="robustness_threshold", palette="deep", alpha=0.8)
    ax.set_xticklabels([f'{float(x.get_text()):.2f}' for x in ax.get_xticklabels()])
    
    legend = ax.legend(title="Robustness Threshold")
    for text in legend.get_texts():
        text.set_text(f'{float(text.get_text()):.2f}')

    ax.set_xlabel("Restaking Degree")
    ax.set_ylabel("Minimum Stake")

    ax.axhline(y=get_start_max_stake(info, multiplier=10), color='k', linestyle='--', label="Max Stake")

    ax.grid(color='gray', linewidth=0.5, alpha=0.5)

    plt.savefig(f"{results_folder_for_figures}/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()


def figure8() -> None:
    n=3
    info = RestakingNetworkInfo(
        num_validators=n,
        num_services=n,
        service_attack_reward=1,
        service_attack_threshold=1/3,
        base_service=(10, 1/2)
    )
    for loss_threshold in [0, 1, 2]:
        build_figure8_stake_graph(f"figure8_y_stake_base_service_{info.base_service[0]}_{info.base_service[1]:.2f}_loss_threshold_{loss_threshold}", info, loss_threshold)


if __name__ == "__main__":
    try:
        os.mkdir(results_folder_for_figures)
    except OSError:
        pass
    
    print("Starting figure 3")
    figure3()

    print("Starting figure 4")
    figure4()

    print("Starting figure 5")
    figure5()

    print("Starting figure 6")
    figure6()

    print("Starting figure 7")
    figure7()

    print("Starting figure 8")
    figure8()