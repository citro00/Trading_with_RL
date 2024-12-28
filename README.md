# **Ambiente di Trading con Deep Reinforcement Learning**

## **Panoramica**
Questo progetto implementa un ambiente di trading per simulare e valutare strategie finanziarie utilizzando il Deep Q-Learning (DQN) e il Q-Learning. I componenti principali includono:

1. **Ambiente di Trading Personalizzato**: Basato su `gym_anytrading`, esteso per il trading multi-asset.
2. **Agente DQN**: Agente basato su rete neurale profonda per la presa di decisioni.
3. **Preprocessing dei Dati**: Funzioni per scaricare, pulire e arricchire i dati finanziari da Yahoo Finance.
4. **Pipeline di Addestramento e Valutazione**: Routine automatizzate per addestrare e testare gli agenti su dati di mercato storici.

---

## **Struttura del Progetto**

### **Componenti Principali**

- **CustomStocksEnv**: Un ambiente personalizzato per il trading di più asset.
- **DQNAgent**: Agente basato su Deep Q-Learning per apprendere politiche ottimali.
- **Funzioni Utility**: Strumenti per il download, la pulizia e l'elaborazione dei dati.
- **Script Principale**: Integra tutti i componenti per l'addestramento e la valutazione.

---

## **Flusso di Lavoro**

1. **Preparazione dei Dati**:
   - Scarica i dati storici dei prezzi utilizzando Yahoo Finance.
   - Pulisci e pre-elabora i dati per calcolare metriche come rendimento giornaliero, rendimento cumulativo, SMA e VWAP.
2. **Configurazione dell'Ambiente**:
   - Configura `CustomStocksEnv` con i dati pre-elaborati e i parametri di trading.
3. **Inizializzazione dell'Agente**:
   - Inizializza l'agente DQN o Q-Learning in base all'input dell'utente.
4. **Addestramento**:
   - Addestra l'agente per più episodi, monitorando metriche come profitto e ricompensa.
5. **Valutazione**:
   - Testa l'agente su dati non visti e visualizza la performance di trading.

---

## **Formulazioni Matematiche**

### **Calcolo della Ricompensa**

La funzione di ricompensa valuta la redditività di ogni azione:

- **Ricompensa di Vendita**:
  $$R_{sell} = \log\left(\frac{P_t}{P_{t_{last}}}\right) + c \text{ se } P_t > P_{t_{last}} \text{ altrimenti } \log\left(\frac{P_t}{P_{t_{last}}}\right) - c$$

- **Ricompensa di Acquisto**:
  $$R_{buy} = \log\left(\frac{P_{t_{last}}}{P_t}\right) + c \text{ se } P_t < P_{t_{last}} \text{ altrimenti } \log\left(\frac{P_{t_{last}}}{P_t}\right) - c$$

- **Ricompensa di Mantenimento**:
  $$R_{hold} = \log\left(\frac{P_t}{P_{t_{last}}}\right) + c \text{ se } P_t > P_{t_{last}} \text{ altrimenti } \log\left(\frac{P_t}{P_{t_{last}}}\right) - c$$

Dove:
- $P_t$: Prezzo al tick corrente.
- $P_{t_{last}}$: Prezzo all'ultimo trade.
- $c$: Costante di bias per incentivare azioni positive.

### **Regola di Aggiornamento del Q-Learning**

Per uno stato \(s\), un'azione \(a\) e una ricompensa \(r\):
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_a Q(s', a') - Q(s, a) \right]$$

Dove:
- $\\alpha$: Tasso di apprendimento.
- $\\gamma$: Fattore di sconto.
- $s'$: Stato successivo.
- $a'$: Azione ottimale nello stato successivo.

### **Funzione di Perdita del DQN**

La perdita è calcolata come:
$$L(\theta) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

Dove $\\theta$ sono i parametri del modello online e $\\theta^-$ sono i parametri del modello target.

---

## **Dettagli dell'Implementazione**

### **Ambiente**
Il `CustomStocksEnv` è progettato per:
- Gestire più asset.
- Fornire uno spazio di osservazione con dati normalizzati di prezzo e volume.
- Calcolare le ricompense basate sulle azioni di trading.

### **Architettura del DQN**

Il DQN è composto da:
- Input: Caratteristiche dello stato osservato.
- Due layer nascosti con attivazioni ReLU.
- Output: Q-valori per ogni azione.

$$\text{Q-valori: } Q(s, a) \approx \text{Rete Neurale}(s; \theta)$$

---

## **Istruzioni per l'Uso**

1. **Installa le Dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Esegui lo Script Principale**:
   ```bash
   python main.py
   ```

3. **Seleziona il Tipo di Agente**:
   Scegli tra `DQN` o `QL` durante l'esecuzione.

---

## **Output di Esempio**

- **Metriche di Addestramento**:
  - Profitto totale e ricompensa.
  - Andamento della perdita durante l'addestramento.

- **Valutazione**:
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
