# Zadanie 2: Serwowanie modelu jako API (BentoML)

## Cel projektu
Celem zadania było wystawienie wytrenowanego w Zadaniu 1 modelu klasyfikacji nowotworu piersi (**Breast Cancer Wisconsin Dataset**) jako usługi REST API. Do realizacji zadania wykorzystano framework **BentoML**, który pozwala na łatwe pakowanie modeli PyTorch i tworzenie wysokowydajnych serwisów inferencyjnych.

## Technologie i wymagania
* **Model:** MLP (Multilayer Perceptron) z Zadania 1.
* **Framework Serwujący:** BentoML.
* **Język:** Python 3.11.
* **Biblioteki:** `bentoml`, `torch`, `pydantic>=2.0`, `numpy`, `requests`.

## Struktura plików zadania
```text
src/
├── export_model.py    # Skrypt eksportujący model z formatu .ckpt do BentoML Store
├── service.py         # Definicja serwisu API (klasa BreastCancerService)
└── client.py          # Klient testowy wysyłający zapytania HTTP POST
```

## Instrukcja uruchomienia

### 1. Eksport modelu
Najpierw należy przenieść wytrenowany model do magazynu modeli BentoML:
```bash
python src/export_model.py
```
*Skrypt ten ładuje najlepszy checkpoint (`.ckpt`), wyodrębnia wagę modelu PyTorch i rejestruje go pod nazwą `breast_cancer_mlp`.*

### 2. Uruchomienie serwisu API
Aby wystartować serwer inferencyjny na lokalnym porcie 3000, użyj komendy:
```bash
bentoml serve src.service:BreastCancerService
```
Po uruchomieniu, pod adresem `http://localhost:3000` dostępny jest interfejs **Swagger UI**, umożliwiający interaktywne testowanie endpointów.

## Przykład zapytania (Client)
Do przetestowania usługi przygotowano skrypt `src/client.py`, który wysyła losową próbkę 30 cech pacjenta w formacie JSON i odbiera predykcję.

**Kod klienta (`src/client.py`):**
```python
import requests
import numpy as np

url = "http://localhost:3000/predict"
sample_data = np.random.rand(1, 30).tolist()
payload = {"input_data": sample_data}

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Predykcja: {response.json()}")
```

Można również użyć komendy **cURL**:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"input_data": [[0.5, 0.2, 0.1, 0.9, ...]]}' \
     http://localhost:3000/predict
```

## Potwierdzenie działania (Output)
Usługa poprawnie przyjmuje zapytania i zwraca wyniki w formacie JSON.

**Logi serwera:**
```text
[INFO] [cli] Starting production HTTP BentoServer from "src.service:BreastCancerService" listening on http://localhost:3000
[INFO] [entry_service:BreastCancerService:1] Service BreastCancerService initialized
[INFO] [entry_service:BreastCancerService:1] 127.0.0.1 (method=POST, path=/predict) status=200
```

**Odpowiedź klienta:**
```text
Wysyłam dane pacjenta do http://localhost:3000/predict...
Sukces! Serwer odpowiedział:
Wynik predykcji (0 = łagodny, 1 = złośliwy): [0]
```

---

Jestem gotowy na polecenie do **zadania 3**! Prześlij je, a przeanalizuję je i zaplanuję kolejne kroki.
