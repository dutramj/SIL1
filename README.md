# VAHSimulator - Simulador de Veículo Acelerador

Um simulador de dinâmica de voo para veículos aceleradores, desenvolvido em Python.

## Sobre o Projeto

O VAHSimulator modela a dinâmica completa de um veículo acelerador em desenvolvimento, incluindo:

- Dinâmica de voo em 6 graus de liberdade (6-DOF)
- Sistemas de navegação inercial (INS) e GNSS
- Controle de atitude com TVC, RCS e superfícies aerodinâmicas
- Modelagem de perturbações atmosféricas e ambientais
- Aquecimento aerodinâmico em regime de voo

## Estrutura do Projeto

```
vahsimulator/
│
├── datasets/                # Look-Up tables contendo dados aerodinâmicos e de atmosfera
├── mission/                 # Dados sobre o plano de voo, as fases dos estágios e os parâmetros dos subsistemas
├── records/                 # Dados gravados das simulações
├── scripts/                 # Scripts auxiliares para rodar simulação e plotar gráficos
├── tests/                   # Testes unitários
└── vahsimulator.py          # Implementação do sistema e seus subsistemas
```

## Instalação

Baixar e instalar VS Code.

O projeto também demanda Python 3.10.8 e PyPoetry. Devido à necessidade de permissão de administrador, é necessário solicitar a instalação do Python e PyPoetry à TI.

Em seguida, é necessário criar o ambiente virtual:
```
poetry lock
poetry install
```

Criar `settings.json` na pasta `.vscode` (caso não existam) contendo:
```
{
    "python-envs.defaultEnvManager": "ms-python.python:poetry",
    "python-envs.defaultPackageManager": "ms-python.python:poetry"
}
```

## Como Usar
Abrir arquivo `tests\run_simulation.py` e pressionar botao "Run Python file" (pequeno botao com símbolo de triãngulo para direita tipo botão Play). Durante um total de cerca de 30 segundos (dependendo do poder computacional do hardware): o código será compilado; uma simulação será executada; e aparecerá uma janela contendo vários gráficos sobre a trajetória do voo

Alternativamente:
```bash
python scripts\run_simulation.py
```

### Configuração de Parâmetros
Edite `mission/rato_mission_3dof_skip_glide.yaml` para ajustar parâmetros do veículo e sensores, ou outro YAML de fato apontado dentro de `scripts\run_simulation`.

### Visualização
O `scripts\run_simulation` automaticamente mostra gráficos.

Alternativamente, duas trajetórias podem ser comparadas executando `scripts\compare_results.py`.

Além disso, há mais gráficos alternativos:
```bash
python plots_vah_3D.py      # Visualização 3D
python plots_vah_globe.py   # Globo terrestre
```

## Saídas

### Dados Gravados
- **Arquivo CSV**: `records/simulation_data.csv` com todos os dados da simulação

### Gráficos Gerados
- Trajetória e posição geodésica
- Atitude e velocidades angulares
- Forças e momentos aerodinâmicos (regime de voo)
- Performance dos sistemas de controle
- Estados dos sensores (IMU, GNSS)
- Dados de navegação e estimação
- Parâmetros de propulsão e aquecimento de voo
- Número de Mach e propriedades do escoamento
- e outros mais

## Performance
- Frequência: 1000 Hz
- Tempo de simulação: ~80 segundos

## Referências

- **BRYSON, Arthur Earl.** Control of spacecraft and aircraft. Princeton: Princeton University Press, 1994.
- **DA SILVA, Adolfo Graciano.** ANALISE E PROJETO DO SISTEMA DE CONTROLE DE ATITUDE DO VEÍCULO LANCADOR DE SATÉLITE (VLS). 2014.
- **BEARD, Randal W.; MCLAIN, Timothy W.** Small unmanned aircraft: Theory and practice. Princeton university press, 2012.
- **ZIPFEL, Peter H.** Modeling and simulation of aerospace vehicle dynamics. AIAA, 2007.
- **Jaggers, R.** Multistage Linear Tangent Ascent Guidance as Baselined for the Space Shuttle Vehicle, NASA MSC, Internal Note MSC-EG-72-39, June 1972.
- **Jaggers, R.** An Explicit Solution to the Exo-atmospheric Powered Flight Guidance and Trajectory Optimization Problem for Rocket Propelled Vehicles, AIAA Paper, Vol. 1051, 1977.
- **U.S. Standard Atmosphere, 1976,** NOAA, NASA, USAF.
- **Wikipedia.** Stagnation temperature. https://en.wikipedia.org/wiki/Stagnation_temperature
- **Wikipedia.** Proportional navigation. https://en.wikipedia.org/wiki/Proportional_navigation

## 👥 Authors & Acknowledgments

- **Copyright**: ©  - All rights reserved
- **Original Development**: 
- **Portability & Maintenance**: GNC Team
- **Special Thanks**: Aerodynamics and GNC teams for validation data

## 📞 Support

For technical support and questions:
- **Primary Contact**: GNC Team
- **Issues**: Use GitLab issue tracker for bug reports
- **Documentation**: Check inline code comments and technical documentation
- **Internal Support**: Contact GNC team directly for implementation assistance
---

**Projeto em desenvolvimento** - 
