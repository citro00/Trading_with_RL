from enum import Enum 

class Action(Enum):
    """
    Enumerazione delle possibili azioni di trading.
    Attributi:
        Sell (int): Azione per vendere gli asset posseduti.
        Buy (int): Azione per acquistare nuovi asset.
        Hold (int): Azione per mantenere la posizione attuale senza effettuare transazioni.
    """
    Sell = 0
    Buy = 1
    Hold = 2
