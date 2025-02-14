import numpy as np

from qiskit import QuantumCircuit, QuantumRegister

def sort_all_tsp(H, mode, print_info=True):
    """
    Order a Hamiltonian group according to the TSP heuristic    """
    
    tspOrder = orderGroupForGateCancellation([term[0] for term in H], mode,
                                              print_info=print_info)
    
    sortedH = [(tspOrder[i], H[i][1]) for i in range(len(H))]
    return sortedH

#Heuristic solver for TSP using 2-opt algorithm from scratch
def calculate_total_distance(tour, distance_matrix):
    """Calculate the total distance of a given tour."""
    return sum(distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + distance_matrix[tour[-1], tour[0]]

def two_opt(distance_matrix, initial_tour=None, max_iterations=1000):
    """Perform the 2-opt algorithm to optimize a given tour."""
    num_cities = len(distance_matrix)
    
    # Initialize tour
    if initial_tour is None:
        tour = list(range(num_cities))  # Default: sequential order
    else:
        tour = initial_tour[:]
    
    best_distance = calculate_total_distance(tour, distance_matrix)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        for i in range(1, num_cities - 1):
            for j in range(i + 1, num_cities):
                if j - i == 1:  # No point in swapping adjacent edges
                    continue
                
                # Create new tour by reversing segment between i and j
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_distance = calculate_total_distance(new_tour, distance_matrix)
                
                # If the new tour is better, update the current tour
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
        iteration += 1
    
    return tour, best_distance

def orderGroupForGateCancellation(group, mode, print_info=True):
    """
    Orders terms within a group (clique) such that the number of possible gate
    cancellations is maximized.

    Uses the genetic_algorithm supplied by mlrose to produce a Travelling
    Salesperson (TSP) path through the graph.

    Produce a Hamiltonian Cycle through the graph by deleting the most expensive
    edge in the TSP path.

    Parameters
    ----------
    group : List(str)
        a clique of Pauli strings
    print_info : bool
        print extra info on Hamiltonian Cycles found

    Returns
    -------
    List(str)
        a clique of Pauli strings where the ordering of the strings maximizes
        gate cancellations
    """
    if len(group) == 1:
        if print_info:
            print('Single term in group, returning: ', list(group))
        return list(group)

    # # Find a TSP path using the mlrose module
    # group = list(group)
    # mlrose_distances, distance_matrix = _pairwise_distances(group, mode=mode)
    # # problem = mlrose.TSPOpt(length=len(group), distances=mlrose_distances)
    # problem = TSPOpt(length=len(group), distances=mlrose_distances)
    # mlrose_best_state, mlrose_best_fitness  = mlrose.genetic_alg(problem)

    # Find a TSP path
    group = list(group)
    distance_matrix = _pairwise_distances(group, mode=mode)
    best_path, best_path_length = two_opt(distance_matrix)

    # Use the distance_matrix to break the most expensive edge in the TSP path,
    # giving a Hamiltonian Cycle with lower overall distance
    path_distances = []
    for i in range(len(best_path)):
        # select a node in the path
        this_node = best_path[i]
        # select the next node in the path
        if i == len(best_path)-1:
            # if this_node is the last node, loop back to the beginning
            next_node = best_path[0]
        else:
            next_node = best_path[i+1]
        # add the distance from this_node to next_node to path_distances
        path_distances.append(distance_matrix[max(this_node,next_node),
                                              min(this_node,next_node)])

    # find the most expensive edge
    largest_dist_seen = -1
    for i, dist in enumerate(path_distances):
        if dist >= largest_dist_seen:
            largest_dist_seen = dist
            expensive_edge = i

    # given the most expensive edge, slice the best_path
    best_path = list(best_path)
    best_state = best_path[expensive_edge+1:] + best_path[:expensive_edge+1]
    best_fitness = np.sum(path_distances) - path_distances[expensive_edge]

    # Use best_state to order the terms within the group
    ordered_group = [group[i] for i in best_state]

    if print_info:
        print(' '*3,'original order was %s' % group)
        print(' '*3,'new order is %s' % ordered_group)
        print(' '*3,'total Hamiltonian Cycle distance is %s' % best_fitness)

    return ordered_group

def _pairwise_distances(group, mode='star'):
    """Pairwise CNOT distances between Pauli strings in group, used for TSP heuristic."""
    distance_matrix = np.zeros((len(group),len(group)), dtype=int)
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            term1, term2 = group[i], group[j]
            if mode == 'star':
                cur_distance = _star_distance(term1, term2)
            elif mode == 'ladder':
                cur_distance = _ladder_distance(term1, term2)
            # We also store the distances in a lower-triangular matrix
            distance_matrix[j,i] = cur_distance
    return distance_matrix

def _ladder_distance(term1, term2):
    """naive ladder CNOT distance between two Pauli strings, used for TSP Heuristic."""
    assert len(term1) == len(term2), '%s and %s have different lengths' % (term1, term2)
    assert all([char in ['I', 'X', 'Y', 'Z'] for char in term1]), '%s has non IXYZ chars' % term1
    assert all([char in ['I', 'X', 'Y', 'Z'] for char in term2]), '%s has non IXYZ chars' % term2

    # First calculate the total required CNOTs, then deduct the cancellation.
    # For each Pauli string, total required CNOTs equals the number of non-I terms minus 1.
    # Edge case happens when there is zero or one non-I character, already handled
    total_non_I_term1 = 0
    total_non_I_term2 = 0
    term1_CNOT = 0
    term2_CNOT = 0
    for i in range(len(term1)):
        if (term1[i] != 'I'):
            total_non_I_term1 += 1
        if (term2[i] != 'I'):
            total_non_I_term2 += 1
    if total_non_I_term1 > 1:
        term1_CNOT = total_non_I_term1 - 1
    if total_non_I_term2 > 1:
        term2_CNOT = total_non_I_term2 - 1

    # Next do deduction
    # If the right most and 2nd right most characters are the same, then we can cancel the first outer layer CNOT pair.
    # Cancel layer by layer after, stop when characters start to differ.
    # Start iterating from the right most Pauli character.
    # Edge case happens when there is zero or one same character, already handled
    same_count = 0
    CNOT_reduction = 0
    for i in range(len(term1)):
        reverse_iterator = len(term1) - 1 - i
        # print(reverse_iterator)
        if term1[reverse_iterator] != term2[reverse_iterator]:
            # if there is any different Pauli characters, break the loop
            break
        elif (term1[reverse_iterator] != 'I' and term2[reverse_iterator] != 'I'):
            # for other non-I same character pairs, increment the same_count by 1
            same_count += 1
    if same_count > 1:
        CNOT_reduction = (same_count - 1) * 2
    # elif same_count == 1 or same_count == 0, CNOT_reduction stays 0

    # print('term1 has CNOT # = ', term1_CNOT)
    # print('term2 has CNOT # = ', term2_CNOT)
    # Increase all ladder distances by 1, because some terms will have
    # a distance = 0 which is really good, but mlrose requires that
    # all distances are > 0
    return term1_CNOT + term2_CNOT - CNOT_reduction + 1

def _star_distance(term1, term2):
    """star + ancilla CNOT distance between two Pauli strings, used for TSP heuristic."""
    assert len(term1) == len(term2), '%s and %s have different lengths' % (term1, term2)
    assert all([char in ['I', 'X', 'Y', 'Z'] for char in term1]), '%s has non IXYZ chars' % term1
    assert all([char in ['I', 'X', 'Y', 'Z'] for char in term2]), '%s has non IXYZ chars' % term2

    # edge case: if there is only zero or one none-I term (e.g. IIIIXIII) in either of the two terms
    # then at least one term have no CNOT gate, and therefore nothing to cancel
    # (this is very unlikely, but under the star + ancilla implementation this could create a little unnecessary CNOT overhead)

    # general case
    # According to Eq.14 in https://arxiv.org/pdf/2001.05983.pdf
    # total of 4 x 4 = 16 XYZI letter combinations
    # for XY, XZ, YX, YZ, ZX, ZY, 2 CNOTs will incur (distance + 2)
    # for IX, IY, IZ, XI, YI, ZI, 1 CNOT will incur (distance + 1)
    # for XX, YY, ZZ everything cancel perfectly (distance + 0)
    # for II, nothing needed (distance + 0)
    distance = 0
    for i in range(len(term1)):
        if term1[i] != term2[i]:
            distance += 2
        if (term1[i] == 'I' and term2[i] == 'X') or (term1[i] == 'I' and term2[i] == 'Y') or (term1[i] == 'I' and term2[i] == 'Z') or (term1[i] == 'X' and term2[i] == 'I') or (term1[i] == 'Y' and term2[i] == 'I') or (term1[i] == 'Z' and term2[i] == 'I'):
            distance -= 1
    return distance

def gen_circuit(H, num_qubits, t: float) -> QuantumCircuit:
    
    """
    Create a circuit implementing the quantum dynamics simulation
    Parameters
    ----------
    H: list
        Hamiltonian
    num_qubits : int
    t : float
        time of evolution
    Returns
    -------
    QuantumCircuit
        QuantumCircuit object with width = nq
    """

    # convert the sorted Hamiltonian into a bitstring representation    
    H_bitstr = get_H_bitstr(H)
    iterate_bitstr = trotter_suzuki(H_bitstr, t)

    # print('H bitstring representation:\n', H_bitstr)

    # create a QuantumCircuit object
    qr = QuantumRegister(num_qubits)
    circ = QuantumCircuit(qr)    
    iterate_circuit = construct_iterate_circuit(qr, iterate_bitstr)
    
    circ.append(iterate_circuit, qr)

    return circ

def get_H_bitstr(H):
    """
    Produce a bitstring representation of the Hamiltonian where
    H = [(a_j, S_j); j = 1,...,m]
    with
    a_j = the coefficient of term j
    S_j = (S_Xj, S_Yj, S_Zj) a vector giving the positions of the Pauli
          matrices within term j
    """
    
    H_bitstr = []
    for group in H:
        for term in group:
            paulistr, a_j = term
            S_Xj = [i for i, pauli in enumerate(paulistr[::-1]) if pauli == "X"]
            S_Yj = [i for i, pauli in enumerate(paulistr[::-1]) if pauli == "Y"]
            S_Zj = [i for i, pauli in enumerate(paulistr[::-1]) if pauli == "Z"]
            S_j = (S_Xj, S_Yj, S_Zj)
            H_bitstr += [(S_j, np.real(a_j))]
    
    return H_bitstr

def trotter_suzuki(H_bitstr, dt: float):
    """
    Construct a bitstring representation of a Trotter-Suzuki iterate using
    the decomposition given in Eqn 4.98 in Nielsen & Chuang (2010)    Parameters
    ----------
    H : bitstring representation of Hamiltonian
    dt : float
        the timestep argument passed to the Trotter-Suzuki formula    Returns
    -------
    bitstring representation of a single TS iterate
    """
    return [(*hterm, dt) for hterm in H_bitstr]

def construct_iterate_circuit(qreg: QuantumRegister, H_bitstr) :
    """
    Create a circuit which implements a single iteration of Trotter-Suzuki
    Parameters
    ----------
    qreg : QuantumRegister
        Register of qubits
    ts_bitstr : List[(a_j, S_j, t_j)]
        A bitstring representation of the Hamiltonian
    Returns
    -------
    iter_circ : QuantumCircuit
        A Qiskit QuantumCircuit object corresponding to the given TS bitstr
    """
    iter_circ = QuantumCircuit(qreg)
    for (S_j, a_j, t_j) in H_bitstr:
        iter_circ.compose(compute_to_Z_basis(qreg, S_j), inplace=True)
        iter_circ.compose(apply_phase_shift(qreg, a_j * t_j, S_j), inplace=True)
        iter_circ.compose(uncompute_to_Z_basis(qreg, S_j), inplace=True)
    return iter_circ

def compute_to_Z_basis(qreg: QuantumRegister, S_j: tuple) -> QuantumCircuit:
    """
    Transform all qubits to the Z basis

    Parameters
    ----------
    qreg : QuantumRegister
        Register of qubits
    S_j : ([int],[int],[int])
        Tuple of arrays giving the locations of the X, Y, and Z pauli
        operations respectively

    Returns
    -------
    circ : QuantumCircuit
        Qiskit QuantumCircuit object implementing the correct change of
        basis operators for the given S_J
    """

    circ = QuantumCircuit(qreg)
    Xlocs, Ylocs, Zlocs = S_j

    for loc in Xlocs:
        circ.h(qreg[loc])
    for loc in Ylocs:
        circ.sdg(qreg[loc])
        circ.h(qreg[loc])

    return circ

def uncompute_to_Z_basis(qreg: QuantumRegister, S_j: tuple) -> QuantumCircuit:
    """
    Transform all qubits back to original basis

    Parameters
    ----------
    qreg : QuantumRegister
        Register of qubits
    S_j : ([int],[int],[int])
        Tuple of arrays giving the locations of the X, Y, and Z pauli
        operations respectively

    Returns
    -------
    circ : QuantumCircuit
        Qiskit QuantumCircuit object implementing the correct change of
        basis operators for the given S_J
    """

    circ = QuantumCircuit(qreg)
    Xlocs, Ylocs, Zlocs = S_j

    for loc in Xlocs:
        circ.h(qreg[loc])
    for loc in Ylocs:
        circ.h(qreg[loc])
        circ.s(qreg[loc])

    return circ

def apply_phase_shift(qreg: QuantumRegister, delta_t: float, S_j: tuple) -> QuantumCircuit:
    """
    Simulate the evolution of exp(-i(dt)Z)

    Parameters
    ----------
    qreg : QuantumRegister
        Register of qubits
    delta_t : float
        the length of the next time step in the Hamiltonian evolution
    S_j : ([int],[int],[int])
        Tuple of arrays giving the locations of the X, Y, and Z pauli
        operations respectively
    mode : str
        Indicates which DQS implementation to use: star or ladder
    """
    circ = QuantumCircuit(qreg)
    all_locs = []
    for pauli_loc in S_j:
        for loc in pauli_loc:
            all_locs.append(loc)
    all_locs = sorted(all_locs)

    if len(all_locs) == 0:
        return circ
    
    # lsb is the Least Significant quBit in this term that is acted on with
    # a non-Identity operator
    lsb = all_locs[-1]

    # apply CNOT ladder -> compute parity
    if len(all_locs) != 1:
        # if len(all_locs) == 1 then there is only a single non-Identity
        # operator in this term so no CNOT gates are needed
        for i in range(len(all_locs) - 1):
            ctrl = all_locs[i]
            trgt = all_locs[i + 1]
            circ.cx(qreg[ctrl], qreg[trgt])

    # apply phase shift to the Least Significant quBit (LSB)
    # RZ applies the unitary gate specified in utils/RZdef.py:
    #     RZ(phi) = exp(-i*phi*Z/2)
    # circ.RZ(2*delta_t, qreg[lsb])
    # rz applies the Qiskit Z-rotation
    circ.rz(2 * delta_t, qreg[lsb])

    # apply CNOT ladder -> uncompute parity
    if len(all_locs) != 1:
        for i in reversed(range(len(all_locs) - 1)):
            ctrl = all_locs[i]
            trgt = all_locs[i + 1]
            circ.cx(qreg[ctrl], qreg[trgt])

    return circ