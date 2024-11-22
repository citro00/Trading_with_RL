import numpy as np
def make_state_hashable(state):
    """
    Converte lo stato in una tupla di float per renderlo hashable.
    
    Args:
        state (array-like): Stato corrente dell'ambiente.
    
    Returns:
        tuple: Tupla di float che rappresenta lo stato.
    """
    try:
        # Se lo stato è multi-dimensionale, appiattiscilo
        flat_state = np.array(state).flatten()
        return tuple(float(x) for x in flat_state)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Errore nella conversione dello stato a float: {e}")
