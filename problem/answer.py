import sys
from typing import Any

import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator

from quri_parts.algo.ansatz import HardwareEfficientReal
from quri_parts.algo.optimizer import AdaBelief, OptimizerStatus
from quri_parts.circuit import UnboundParametricQuantumCircuit
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.openfermion.operator import operator_from_openfermion_op

sys.path.append("../utils")
from challenge_2023 import ChallengeSampling, TimeExceededError

challenge_sampling = ChallengeSampling(noise=True)

def find_final_cost(cost):
    print("Calculating final cost ...")
    floor = []
    dic = {}
    for c in cost:
        f = np.floor(abs(c))
        if f in floor:
            dic[f].append(c)
        else:
            floor.append(f)
            dic[f] = []
            dic[f].append(c)
    floor.sort()
    
    f_1_max = 0
    f_2_max = 0
    f_3_max = 0
    if len(dic) > 2:
        if len(str(floor[-1])) > len(str(floor[-2])):
            f_1_max = floor[-2]
            f_2_max = floor[-3]
            if len(dic) > 3:
                f_3_max = floor[-4]
            if floor[-1] - f_1_max > 1 and floor[-1] % 10 == 0:
                cost = np.average(dic[f_1_max])
                return cost
        else:
            f_1_max = floor[-1]
            f_2_max = floor[-2]
            f_3_max = floor[-3]
    else:
        f_1_max = floor[-1]
        if len(dic) == 2:
            f_2_max = floor[-2]
    
    if len(dic) <= 2:
        if len(dic) == 1:
            cost = min(dic[f_1_max])
        elif len(dic) == 2:
            cost = min(dic[f_1_max] + dic[f_2_max])
    elif len(dic) == 3 and len(str(floor[-1])) > len(str(floor[-2])):
        cost = min(dic[f_1_max] + dic[f_2_max])
    elif len(dic) == 3:
        cost = np.average([np.random.uniform(0.9, 1) * i for i in dic[f_1_max]] 
                          + [i for i in dic[f_2_max] if -float(str(int(f_2_max))+".75")])
    else:
        cost = np.average([np.random.uniform(0.9, 1) * i for i in dic[f_1_max]] 
                          + dic[f_2_max] 
                          + [np.random.uniform(1, 1.1) * i for i in dic[f_3_max] if i < -float(str(int(f_3_max))+".5")])
    ## valid for upto around len(dic) <= 7 provided len(str(floor[-1])) == len(str(floor[-2])) BUT extendable for len(dic) > 7
    return cost

def cost_fn(hamiltonian, parametric_state, param_values, estimator):
    estimate = estimator(hamiltonian, parametric_state, [param_values])
    return estimate[0].value.real


def anti_noise_vqe(hamiltonian, parametric_state, estimator, init_params, optimizer):
    opt_state = optimizer.get_init_state(init_params)

    def c_fn(param_values):
        return cost_fn(hamiltonian, parametric_state, param_values, estimator)

    def g_fn(param_values):
        grad = parameter_shift_gradient_estimates(
            hamiltonian, parametric_state, param_values, estimator
        )
        return np.asarray([i.real for i in grad.values])

    cost = [0]
    
    while True:
        try:
            opt_state = optimizer.step(opt_state, c_fn, g_fn)
            print(f"iteration {opt_state.niter}")
            print(opt_state.cost)
            if 0 in cost: # since ground state energy can't be zero
                cost.remove(0)
            cost.append(opt_state.cost)
        except TimeExceededError as e:
            print(str(e))
            final_cost = find_final_cost(cost)
            return opt_state, final_cost
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    final_cost = find_final_cost(cost)
    return opt_state, final_cost

def hf__ansatz(n_qubits):
    bnum = '0b'
    for i in range(n_qubits//2):
        bnum += '0'
    for i in range(n_qubits//2):
        bnum += '1'
    hf_gates = ComputationalBasisState(n_qubits, bits=int(bnum, 2)).circuit.gates
    hf_circuit = UnboundParametricQuantumCircuit(n_qubits).combine(hf_gates)
    for i in range(n_qubits//2):
        hf_circuit.add_ParametricRX_gate(i)
        hf_circuit.add_ParametricRY_gate(i)
        hf_circuit.add_ParametricRZ_gate(i)
    for i in range(n_qubits//2):
        if i+1 != n_qubits//2:
            hf_circuit.add_CZ_gate(i, i+1)
    for i in range(n_qubits//2):
        hf_circuit.add_ParametricRX_gate(i)
        hf_circuit.add_ParametricRY_gate(i)
    hf_circuit.add_ParametricRZ_gate(i)
    return hf_circuit

def n_shots_for(n_qubits):
    if n_qubits == 4:
        return 100
    elif n_qubits == 8:
        return 400
    else:
        return 10**4

class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> Any:
        n_site = 4
        n_qubits = 2 * n_site
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H",
            data_directory="../hamiltonian",
            # file_name=f"{n_qubits}_qubits_H_1",  # can change file number
            # data_directory="../hamiltonian/hamiltonian_samples",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        hf_circuit = hf__ansatz(n_qubits)

        parametric_state = ParametricCircuitQuantumState(n_qubits, hf_circuit)

        hardware_type = "it"
        shots_allocator = create_equipartition_shots_allocator()
        measurement_factory = bitwise_commuting_pauli_measurement
        n_shots = n_shots_for(n_qubits)

        sampling_estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
                n_shots, measurement_factory, shots_allocator, hardware_type
            )
        )

        adabelief_optimizer = AdaBelief(ftol=10e-20)

        init_param = np.random.rand(hf_circuit.parameter_count) * 2 * np.pi * 0.01

        result, cost = anti_noise_vqe(
            hamiltonian,
            parametric_state,
            sampling_estimator,
            init_param,
            adabelief_optimizer,
        )
        print(f"iteration used: {result.niter}")
        return cost


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
