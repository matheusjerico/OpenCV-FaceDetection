
# Detecção facial (imagem) usando OpenCV 
#### Utilizando um modelo de deep learning pré treinado do OpenCV
Parâmetros importantes para utilizar a rede pré treinada:
- .prototxt: arquiteura do modelo
- .caffemodel: peso das camadas do modelo


1.0 Importando bibliotecas


```python
import numpy as np
import cv2
```

1.1 Importando localização da imagem


```python
loc_image = "./images/iron_chic.jpg" 
```

1.2 Importando arquitetura do modelo e pesos


```python
loc_caffemodel = "./import-opencv/res10_300x300_ssd_iter_140000.caffemodel"
loc_prototxt = "./import-opencv/deploy.prototxt.txt"
```

1.3 Carregando modelo e imagem do disco


```python
# Carregando modelo do disco na variável net
net = cv2.dnn.readNetFromCaffe(loc_prototxt, loc_caffemodel)

# Carrega a imagem de entrada e constrói um blob de entrada para a imagem
# Redimensionando para um fixo de 300x300 pixels e depois normalizando-o
image = cv2.imread(loc_image)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

```

2.0 Aplicando detecção facial


```python
net.setInput(blob)
detections = net.forward()
```

2.1 Loop das detecções e criando boxes


```python
for i in range(0, detections.shape[2]):

    # extrair a confiança (isto é, a probabilidade) associada ao predição
    confidence = detections[0, 0, i, 2]

    # filtrar as detecções fracas assegurando que a 'confiança' é maior que a confiança mínima
    if confidence > 0.5:
        
        # Calcula as coordenadas (x, y) da caixa demilitadora para o objeto
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # desenhe a caixa delimitadora do rosto juntamente com o probabilidade
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
```

3. Resultado


```python
# Mostra imagem de saida
cv2.imshow("Output", image)
cv2.waitKey(0)
```




    48


---
---
---

# Detecção facial (video) usando OpenCV
#### Utilizando um modelo de deep learning pré treinado do OpenCV
Parâmetros importantes para utilizar a rede pré treinada:
- .prototxt: arquiteura do modelo
- .caffemodel: peso das camadas do modelo



```python
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2 
```

1. Importando arquitetura e pesos do modelo


```python
loc_caffemodel = "./import-opencv/res10_300x300_ssd_iter_140000.caffemodel"
loc_prototxt = "./import-opencv/deploy.prototxt.txt"
```

1.1. Carregando modelo e imagem do disco


```python
# Carregando modelo do disco na variável net
net = cv2.dnn.readNetFromCaffe(loc_prototxt, loc_caffemodel)
```

1.2. Inicialize o fluxo de vídeo e permita que o sensor da câmera aqueça



```python
vs = VideoStream(src=2).start()
time.sleep(2.0)
```

1.3. Loop


```python
# Loop sobre cada frame
while True:
    # Pega o frame e redimenciona com width de 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    
    # Pegue as dimensões do quadro e converta-o em um blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # passe o blob pela rede e obtenha as predições de detecções
    net.setInput(blob)
    detections = net.forward()
    
    # loop sobre as detecções
    for i in range(0, detections.shape[2]):
        # extrair a confiança (isto é, a probabilidade) associada ao predição
        confidence = detections[0, 0, i, 2]

        # filtrar as detecções fracas assegurando que a 'confiança' é maior que a confiança mínima
        if confidence > 0.5:
            # Calcula as coordenadas (x, y) da caixa demilitadora para o objeto
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # desenhe a caixa delimitadora do rosto juntamente com o probabilidade
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # mostrar o frame de saida
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # para o loop quando digita 'q'
    if key == ord("q"):
        break
 
# limpeza
cv2.destroyAllWindows()
vs.stop()
vs.stream.release()
```


```python

```
