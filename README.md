# 1IADT - Hackaton - Fase 5

---

# Detecção de Materiais Cortantes

A FIAP VisionGuard, empresa de monitoramento de câmeras de segurança, está analisando a viabilidade de uma nova funcionalidade para otimizar o seu software.

O objetivo da empresa é usar de novas tecnologias para identificar situações atípicas e que possam colocar em risco a segurança de estabelecimentos e comércios que utilizam suas câmeras.

Um dos principais desafios da empresa é utilizar Inteligência Artificial para identificar objetos cortantes (facas, tesouras e similates) e emitir alertas para a central de segurança.

A empresa tem o objetivo de validar a viabilidade dessa feature, e para isso será necessário fazer um MVP para detecção supervisionada desses objetos.

---

## Objetivos

* Construir ou buscar um dataset contendo imagens de facas, tesouras e outros objetos cortantes em diferentes condições de ângulo e iluminação.
* Anotar o dataset para treinar o modelo supervisionado, incluindo imagens negativas (sem objetos perigosos) para reduzir falsos positivos.
* Treinar o modelo.
* Desenvolver um sistema de alertas (pode ser e-mail).

---

# Solução Desenvolvida

Este é um sistema baseado em **Flask** e **Ultralytics** **YOLO** que permite detectar objetos cortantes em vídeos e em tempo real via webcam. Ele utiliza o modelo **YOLOv11n,** treinado com 7 mil novas imagens, para detecção, exibindo os resultados de forma interativa via interface web.

Para uma inferência mais rápida e eficiente, recomenda-se a utilização do **PyTorch** para o processamento do modelo YOLO, permitindo um desempenho otimizado e reduzindo o tempo de resposta em aplicações em tempo real.

---

## Instalação do PyTorch

Para saber como instalar no Python e validar a versão do CUDA, acessar o site do [PyTorch](https://pytorch.org/get-started/locally/).

---

## Funcionalidades

* **Upload de vídeos** e detecção automática de objetos cortantes.
* **Detecção via Webcam** em tempo real.
* **Histórico de análises** com visualização e exclusão.

---

## Como instalar e rodar

### 1. Clone o repositório

Abra o terminal e execute:

```bash
git clone https://github.com/lgcarvalho/Hackaton-Fase5.git
```

Execute o comando a seguir para entrar na pasta

```
cd Hackaton-Fase5
```

### 2. Crie um ambiente virtual

Certifique-se de ter o **Python 3.9+** instalado. Depois, execute:

```bash
python -m venv venv
```

Entre no ambiente criado executando:

```
venv\Scripts\activate
```

### 3. Instale as dependências

Instale as depedências executando:

```
pip install -r requirements.txt
```

### 4.  Execute o sistema

Execute o comando a seguir para ir na pasta do sistema:

```bash
cd Sistema
```

Execute o código .py:

```
python app.py
```

O servidor rodará em `http://127.0.0.1:5000`.

---

## Estrutura do projeto

```
Sistema/
│── model/
│   ├── best.pt         # Modelo YOLOv11 treinado
│   ├── bytetrack.yaml  # Configuração do ByteTrack
│── static/
│   ├── detections/     # Armazena imagens das detecções
│── templates/          # HTML do sistema
│── app.py              # Código principal (Flask)
│── requirements.txt    # Dependências do projeto
│── README.md           # Documentação
```
