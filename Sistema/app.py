import os
import cv2
import datetime
import uuid
import threading
import time

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from ultralytics import YOLO

# Define os endpoints válidos para a utilização com a webcam
VALID_WEBCAM_ENDPOINTS = {
    "process_webcam",
    "stop_webcam",
    "video_feed",
    "analises_json",
    "detalhe_popup"
}

# Define o diretório onde as imagens de detecções serão salvas
DETECTIONS_DIR = os.path.join("static", "detections")
os.makedirs(DETECTIONS_DIR, exist_ok=True)  # Cria a pasta, se não existir

# Variáveis globais para armazenar o último frame da webcam, a thread de processamento e as análises realizadas
last_frame = None
webcam_thread = None
webcam_thread_running = False
analises = []

# Inicializa a aplicação Flask
app = Flask(__name__)
app.secret_key = "chave-secreta-para-session-flash"  # Chave para sessões e mensagens flash

# Carrega o modelo YOLO treinado para detecção de objetos
model = YOLO("model/best.pt")

# Rota que rendereiza a página inicial
@app.route("/")
def index():
    return render_template("index.html")  # Renderiza a página inicial

# Rota que recebe um vídeo, processa as detecções, salva a imagem e registra a análise
@app.route("/process_video", methods=["POST"])
def process_video():
    # Verifica se um arquivo de vídeo foi enviado no formulário
    if "video" not in request.files:
        flash("Nenhum arquivo de vídeo foi enviado.")
        
        return redirect(url_for("index"))

    video_file = request.files["video"]
    
    # Verifica se o nome do arquivo não está vazio
    if video_file.filename == "":
        flash("Nenhum arquivo de vídeo foi selecionado.")
        
        return redirect(url_for("index"))

    # Salva o vídeo enviado na pasta "static"
    video_path = os.path.join("static", video_file.filename)
    video_file.save(video_path)

    # Gera um identificador único para essa análise
    analise_id = str(uuid.uuid4())
    detection_index = 0  # Contador de detecções no vídeo
    seen_track_ids = set()  # Conjunto para armazenar IDs de rastreamento já vistos

    # Processa o vídeo utilizando o modelo com rastreamento de objetos
    results_generator = model.track(
        source=video_path,
        tracker="model/bytetrack.yaml",
        persist=True,
        conf=0.5,
        stream=True,
        augment=True, 
        half=True
    )

    # Itera sobre cada frame/resultados do vídeo
    for result in results_generator:
        boxes = result.boxes
        
        # Se não houver detecções, passa para o próximo frame
        if not boxes:
            continue

        frame = result.orig_img  # Obtém o frame original do vídeo

        # Itera sobre cada caixa (detecção) encontrada
        for box in boxes:
            cls_id = int(box.cls[0].item())          # ID da classe detectada
            label = result.names[cls_id]               # Nome da classe (ex.: "objeto_cortante")
            track_id = box.id[0].item() if box.id is not None else None  # ID de rastreamento do objeto

            conf = float(box.conf[0].item())           # Confiança da detecção
            conf_percent = conf * 100.0                # Converte a confiança para porcentagem
    
            # Se o objeto detectado for "objeto_cortante" e ainda não tiver sido registrado
            if track_id is not None and label == "objeto_cortante":
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    detection_index += 1

                    # Desenha um retângulo ao redor da detecção
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Exibe informações da detecção no frame
                    text = f"{label} {conf_percent:.2f}% (ID {int(track_id)})"
                    cv2.putText(frame, text, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Salva a imagem do frame com a detecção marcada
                    filename = f"{analise_id}_{detection_index:06d}.jpg"
                    cv2.imwrite(os.path.join(DETECTIONS_DIR, filename), frame)

    # Remove o arquivo de vídeo após o processamento
    os.remove(video_path)
    
    # Adiciona as informações da análise na lista global
    analises.append({
        "id": analise_id,
        "fonte": "Video",
        "data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "qtd_deteccoes": detection_index,
    })

    flash("Análise de vídeo concluída!")
    
    return redirect(url_for("lista_analises"))

# Função que processa os frames da webcam em um loop contínuo
def webcam_detection_loop():
    global webcam_thread_running, last_frame

    # Abre a webcam (dispositivo 0)
    cap = cv2.VideoCapture(0)
    
    # Se não conseguir acessar a webcam, finaliza a thread
    if not cap.isOpened():
        webcam_thread_running = False
        
        print("Erro ao acessar a webcam.")
        
        return
    
    seen_track_ids = set()  # Conjunto para armazenar IDs já detectados

    # Loop que captura e processa os frames da webcam enquanto a flag estiver True
    while webcam_thread_running:
        ret, frame = cap.read()
        
        # Se não conseguir ler o frame, aguarda um pouco e tenta novamente
        if not ret:
            time.sleep(0.1)
            continue

        deteccao_ativa = False  # Flag que indica se houve alguma detecção no frame

        results = model.track(
            frame, 
            tracker="model/bytetrack.yaml",
            persist=True,
            conf=0.5,
            stream=True,
            augment=True, 
            half=True
        )

        results = list(results)  # Converte o gerador de resultados em uma lista

        if results:
            r = results[0]
            boxes = r.boxes

            # Processa cada detecção encontrada no frame
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = r.names[cls_id]
                track_id = box.id[0].item() if box.id is not None else None
                conf = float(box.conf[0].item()) * 100.0

                # Se o objeto for "objeto_cortante", realiza a marcação
                if label == "objeto_cortante":
                    deteccao_ativa = True

                    # Desenha um retângulo ao redor da detecção
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Exibe informações da detecção no frame
                    text = f"[!] {label} {conf:.2f}% (ID {int(track_id) if track_id else 'N/A'})"
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                    # Se for uma nova detecção, salva o frame e registra a análise
                    if track_id and track_id not in seen_track_ids:
                        seen_track_ids.add(track_id)
                        analise_id = str(uuid.uuid4())
                        filename = f"{analise_id}.jpg"
                        cv2.imwrite(os.path.join(DETECTIONS_DIR, filename), frame)

                        analises.append({
                            "id": analise_id,
                            "fonte": "WebCam",
                            "data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "qtd_deteccoes": 1,
                        })

        # Se houve detecção, desenha uma borda vermelha ao redor do frame para destaque
        if deteccao_ativa:
            border_thickness = 10
            frame[:border_thickness, :] = (0, 0, 255)
            frame[-border_thickness:, :] = (0, 0, 255)
            frame[:, :border_thickness] = (0, 0, 255)
            frame[:, -border_thickness:] = (0, 0, 255)

        # Atualiza o último frame processado (usado para streaming)
        last_frame = frame

    cap.release()  # Libera a webcam quando o loop é encerrado
    print("Webcam encerrada.")

# Rota que processa a webcam em tempo real, verifica se a webcam esta disponivel e inicia a thread de processamento
@app.route("/process_webcam")
def process_webcam():
    # Testa se a webcam está disponível
    test_cap = cv2.VideoCapture(0)
    
    if not test_cap.isOpened():
        test_cap.release()
        
        flash("Nenhuma webcam detectada.")
        
        return redirect(url_for("index"))
    
    test_cap.release()

    global webcam_thread, webcam_thread_running
    
    # Se a thread de processamento da webcam não estiver rodando, inicia-a
    if not webcam_thread_running:
        webcam_thread_running = True
        webcam_thread = threading.Thread(target=webcam_detection_loop, daemon=True)
        webcam_thread.start()

    return render_template("webcam.html")  # Renderiza a página que exibe o feed da webcam

# Rota que para a webcam manualmente
@app.route("/stop_webcam", methods=["POST"])
def stop_webcam():
    global webcam_thread_running

    # Altera a flag para parar o loop da webcam
    webcam_thread_running = False
    
    print("Webcam foi parada manualmente pelo usuário.")
    
    return '', 204  # Retorna uma resposta vazia com status 204 (No Content)

# Rota que fornece o streaming do feed da webcam em tempo real
@app.route("/video_feed")
def video_feed():
    def gen_frames():
        # Gera os frames para o streaming da webcam
        while True:
            if not webcam_thread_running:
                break
            
            if last_frame is not None:
                ret, buffer = cv2.imencode('.jpg', last_frame)
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    # Formata os bytes do frame para transmissão em multipart
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
            time.sleep(0.02)  # Pequena pausa para controlar a taxa de envio dos frames

    # Retorna uma resposta HTTP que transmite o vídeo em tempo real
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Rota que exibe a lista de análises realizadas
@app.route("/lista_analises")
def lista_analises():
    # Renderiza a página que lista as análises realizadas
    return render_template("lista_analises.html", analises=analises)

# Rota que exibe os detalhes de uma análise específica
@app.route("/detalhe/<analise_id>")
def detalhe_analise(analise_id):
    # Busca a análise correspondente ao ID fornecido
    analise = next((a for a in analises if a["id"] == analise_id), None)
    
    if not analise:
        flash("Análise não encontrada.")
        return redirect(url_for("lista_analises"))

    imagens = []
    
    # Coleta todas as imagens relacionadas à análise (arquivos que começam com o ID)
    for fn in os.listdir(DETECTIONS_DIR):
        if fn.startswith(analise_id):
            imagens.append(fn)
            
    imagens.sort()  # Ordena as imagens
    
    return render_template("detalhe_analise.html", analise=analise, imagens=imagens)

# Rota que exclui uma análise específica
@app.route("/delete_analise/<analise_id>", methods=["POST"])
def delete_analise(analise_id):
    # Busca a análise que será excluída
    analise = next((a for a in analises if a["id"] == analise_id), None)
    
    if not analise:
        flash("Análise não encontrada para exclusão.")
        return redirect(url_for("lista_analises"))

    # Remove todos os arquivos de imagem associados à análise
    for fn in os.listdir(DETECTIONS_DIR):
        if fn.startswith(analise_id):
            os.remove(os.path.join(DETECTIONS_DIR, fn))
            
    analises.remove(analise)  # Remove a análise da lista global
    
    flash("Análise excluída com sucesso.")
    
    return redirect(url_for("lista_analises"))

# Rota que simula o envio de e-mail com os dados da análise
@app.route("/enviar_email/<analise_id>", methods=["POST"])
def enviar_email(analise_id):
    # Obtém o e-mail de destino a partir do formulário
    email_destino = request.form.get("email_destino")
    
    # Simula o envio de e-mail com os dados da análise
    flash(f"Simulando envio de e-mail para {email_destino} (análise {analise_id}).")
    
    print(f"[SIMULAÇÃO] Email enviado para {email_destino}.")
    
    return redirect(url_for("lista_analises"))

# Rota que exibe os detalhes de uma análise em um popup
@app.route("/detalhe_popup/<analise_id>")
def detalhe_popup(analise_id):
    # Busca a análise pelo ID fornecido
    analise = next((a for a in analises if a["id"] == analise_id), None)
    
    if not analise:
        flash("Análise não encontrada.")
        
        return redirect(url_for("lista_analises"))

    imagens = []
    # Coleta as imagens associadas à análise
    for fn in os.listdir(DETECTIONS_DIR):
        if fn.startswith(analise_id):
            imagens.append(fn)
            
    imagens.sort()  # Ordena as imagens
    
    return render_template("detalhe_popup.html", analise=analise, imagens=imagens)

# Rota que fornece as análises como JSON
@app.route("/analises_json")
def analises_json():
    return jsonify(analises)

# Inicializa o servidor Flask
if __name__ == "__main__":
    app.run(debug=True)  # Inicia o servidor Flask em modo debug