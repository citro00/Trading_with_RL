# **Progetto di Intelligenza Artificiale: Trading con DQN e Q-Learning**

## **Indice**

1. [Panoramica](#panoramica)
2. [Struttura del Progetto](#struttura-del-progetto)
3. [Agenti Implementati](#agenti-implementati)
   - [Agente DQN](#agente-dqn)
   - [Agente Q-Learning](#agente-q-learning)
4. [Formulazioni Matematiche](#formulazioni-matematiche)
   - [Funzione di Ricompensa](#funzione-di-ricompensa)
   - [Regola di Aggiornamento del Q-Learning](#regola-di-aggiornamento-del-q-learning)
   - [Funzione di Perdita del DQN](#funzione-di-perdita-del-dqn)
5. [Flusso di Lavoro](#flusso-di-lavoro)
6. [Istruzioni per l'Uso](#istruzioni-per-luso)
7. [Output di Esempio](#output-di-esempio)
8. [Riferimenti](#riferimenti)
9. [Contatti](#contatti)

---

## **Panoramica**

Questo progetto implementa un ambiente di trading per simulare e valutare strategie finanziarie utilizzando due approcci principali di reinforcement learning:

- **Deep Q-Learning (DQN)**
- **Q-Learning (QL)**

I componenti principali includono:

1. **Ambiente di Trading Personalizzato**: Basato su gym_anytrading, esteso per il trading multi-asset.
2. **Agenti di Reinforcement Learning**: Implementazioni sia di DQN che di Q-Learning tabulare.
3. **Preprocessing dei Dati**: Funzioni per scaricare, pulire e arricchire i dati finanziari da Yahoo Finance.
4. **Pipeline di Addestramento e Valutazione**: Routine automatizzate per addestrare e testare gli agenti su dati di mercato storici.

---

## **Struttura del Progetto**

### **Componenti Principali**

- **CustomStocksEnv**: Ambiente di trading personalizzato per multi-asset.
- **DQNAgent**: Agente basato su Deep Q-Learning per apprendere politiche ottimali.
- **QLAgent**: Agente basato su Q-Learning tabulare per apprendimento incrementale.
- **Funzioni Utility**: Strumenti per il download, la pulizia e l'elaborazione dei dati.
- **Script Principale**: Integra tutti i componenti per l'addestramento e la valutazione.

---

## **Agenti Implementati**

### **Agente DQN**
L'agente DQN utilizza una rete neurale profonda per approssimare i valori Q. Questo approccio consente di affrontare spazi di stato continui e di apprendere politiche ottimali basate su esperienze storiche.

**Caratteristiche Principali:**
- Rete neurale composta da due layer nascosti con attivazioni ReLU.
- Aggiornamento tramite una funzione di perdita MSE (Mean Squared Error).
- Utilizzo di replay buffer e rete target per stabilità.

### **Agente Q-Learning**
L'agente Q-Learning è un approccio tabulare che utilizza una tabella Q per stimare i valori delle azioni in base agli stati discreti. È particolarmente adatto per ambienti con spazi di stato discreti e facilmente discretizzabili.

**Caratteristiche Principali:**
- Tabella Q inizializzata dinamicamente.
- Supporto per esplorazione ε-greedy con decadimento di epsilon.
- Discretizzazione degli stati basata su intervalli definiti.
- Metriche dettagliate per l'addestramento e la valutazione.

---

## **Formulazioni Matematiche**

### **Funzione di Ricompensa**

La funzione di ricompensa valuta la redditività di ogni azione:

$R(s_t, a_t) = \Delta P_t - \text{penalita}_h - \text{penalita}_{\text{drawdown}} - \text{penalita}_{\text{transazione}}$

Dove:
- $\Delta P_t$: variazione del valore del portafoglio.
- Penalità per inattività, drawdown, e costi di transazione.

**Delta del portafoglio:**

$$\Delta P_t = V_t - V_{\tau}$$

Dove:
- $V_t = \text{cash}_t + (\text{azioni}_t \cdot \text{prezzo}_t)$: valore attuale del portafoglio.
- $V_{\tau}$: valore del portafoglio all'ultimo trade.

**Penalità di inattività:**

h = $$\lambda_h (\beta_a \cdot n_{\text{azioni}} + \beta_i \cdot n_{\text{hold}})$$


**Penalità di drawdown:**

$$
\lambda_d \cdot \begin{cases} 
\alpha \cdot \frac{V_{\max} - V_t}{V_{\max}}, & \text{se } \frac{V_{\max} - V_t}{V_{\max}} > 0.5 \\
0, & \text{altrimenti}
\end{cases}
$$

**Penalità di transazione:**

$$\lambda_t \cdot 0.05 \cdot \text{prezzo}_t$$

### **Regola di Aggiornamento del Q-Learning**

Per uno stato $s$, un'azione $a$ e una ricompensa $r$:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_a Q(s', a') - Q(s, a) \right]$$

Dove:
- $\alpha$: Tasso di apprendimento.
- $\gamma$: Fattore di sconto.
- $s'$: Stato successivo.
- $a'$: Azione ottimale nello stato successivo.

### **Funzione di Perdita del DQN**

La perdita è calcolata come:
$$L(\theta) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

Dove $\theta$ sono i parametri del modello online e $\theta^-$ sono i parametri del modello target.

---

## **Flusso di Lavoro**

1. **Preparazione dei Dati:**
   - Scarica i dati storici dei prezzi utilizzando Yahoo Finance.
   - Pulisci e pre-elabora i dati per calcolare metriche come rendimento giornaliero, rendimento cumulativo, SMA e VWAP.

2. **Configurazione dell'Ambiente:**
   - Configura CustomStocksEnv con i dati pre-elaborati e i parametri di trading.

3. **Inizializzazione dell'Agente:**
   - Scegli tra l'agente DQN e Q-Learning in base al tipo di addestramento desiderato.

4. **Addestramento:**
   - Addestra l'agente per più episodi, monitorando metriche come profitto e ricompensa.

5. **Valutazione:**
   - Testa l'agente su dati non visti e visualizza la performance di trading.

---

## **Istruzioni per l'Uso**

1. **Installa le Dipendenze:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Esegui lo Script Principale:**
   ```bash
   python main.py
   ```

3. **Seleziona il Tipo di Agente:**
   Durante l'esecuzione, scegli tra DQN o QL.

---

## **Output di Esempio**

- **Metriche di Addestramento:**
  - Profitto totale e ricompensa.
  - Andamento della perdita durante l'addestramento.

- **Valutazione:**
  - Performance di trading visualizzata come segnali di acquisto/vendita sul grafico dei prezzi.
  - Profitto finale e ROI.

---

## **Riferimenti**

1. [Gym-Anytrading](https://github.com/AminHP/gym-anytrading)
2. [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
3. [Yahoo Finance API](https://pypi.org/project/yfinance/)

---

## **Contatti**

Per domande o contributi, apri un issue o invia una pull request.

