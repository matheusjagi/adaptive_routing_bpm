import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "BPI_Challenge_2017.xes.gz")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")
LOG_PARQUET_PATH = os.path.join(PROCESSED_DIR, "bpic2017_log.parquet")

DECISION_ACTIVITY = "A_Validating"
DECISION_BRANCHES = ["O_Returned", "O_Accepted", "W_Validate application", "A_Incomplete", "A_Denied"]

# Ramos genuinamente roteáveis (vias de remediação); O_Accepted/A_Denied são desfechos
# da análise de crédito, fora do controle de uma política de encaminhamento.
ROUTABLE_BRANCHES = ["O_Returned", "A_Incomplete", "W_Validate application"]

WARMUP_DAYS = 60          # período inicial usado para fixar a política estática
RETRAIN_DAYS = 30         # cadência de retreino da baseline centralizada
ORACLE_WINDOW_DAYS = 14   # meia-janela do estimador contrafactual (só para avaliação)

# Split temporal para seleção de hiperparâmetros do DV (afinar antes, congelar e testar depois)
TUNE_TEST_SPLIT = "2016-07-01"
TU_DELTA_DAYS = 14        # coorte dos triggered updates
TU_BETA = 1.0             # fator do sinal de casos abertos

CASE_ATTRS = ["case:LoanGoal", "case:ApplicationType", "case:RequestedAmount"]

TERMINAL_STATES = ["A_Pending", "A_Cancelled", "A_Denied"]
CENSORING_BUFFER_DAYS = 41  # p99 do custo das decisões (~40.7 dias)

RANDOM_SEED = 42
