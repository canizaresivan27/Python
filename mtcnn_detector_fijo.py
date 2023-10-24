#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Librerías
# ==============================================================================
import numpy as np
import os
import cv2
import torch
import warnings
import pandas as pd
import logging
import platform
import glob
import PIL
import facenet_pytorch
import time
from datetime import datetime
from typing import Union
from PIL import Image
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import requests


#******************************
entrenar = False
band = True

IP_esp32 = '192.168.80.253'   # Dirección IP asociada a la cámara.
# IP_esp32 = '192.168.40.6'   # Dirección IP asociada a la cámara.
# IP_API = 'http://192.168.25.5:8000'    # Dirección IP de la API.
IP_API = 'http://192.168.80.254:8000'    # Dirección IP de la API.

#******************************

warnings.filterwarnings('ignore')

# Funciones para la detección, extracción, embedding, identificación y gráficos
# ==============================================================================
def detectar_caras(imagen: Union[PIL.Image.Image, np.ndarray],
                   detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                   keep_all: bool        = True,
                   min_face_size: int    = 160,
                   thresholds: list      = [0.6, 0.7, 0.7],
                   device: str           = None,
                   min_confidence: float = 0.9,
                   fix_bbox: bool        = True,
                   verbose               = False)-> np.ndarray:
   
    
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser `np.ndarray, PIL.Image`. Recibido {type(imagen)}."
        )
    
    # Detección de caras
    # --------------------------------------------------------------------------
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen).astype(np.float32)
        
    bboxes, probs = detector.detect(imagen, landmarks=False)
    
    if bboxes is None:
        bboxes = np.array([])
        probs  = np.array([])
    else:
        # Se descartan caras con una probabilidad estimada inferior a `min_confidence`.
        bboxes = bboxes[probs > min_confidence]
        probs  = probs[probs > min_confidence]


    # Corregir bounding boxes
    #---------------------------------------------------------------------------
    # Si alguna de las esquinas de la bounding box está fuera de la imagen, se
    # corrigen para que no sobrepase los márgenes.
    if len(bboxes) > 0 and fix_bbox:       
        for i, bbox in enumerate(bboxes):
            if bbox[0] < 0:
                bboxes[i][0] = 0
            if bbox[1] < 0:
                bboxes[i][1] = 0
            if bbox[2] > imagen.shape[1]:
                bboxes[i][2] = imagen.shape[1]
            if bbox[3] > imagen.shape[0]:
                bboxes[i][3] = imagen.shape[0]

    # Información de proceso
    # ----------------------------------------------------------------------
    if verbose:
        print("----------------")
        print("Imagen escaneada")
        print("----------------")
        print(f"Caras detectadas: {len(bboxes)}")
        # print(f"Corrección bounding boxes: {ix_bbox}")
        print(f"Coordenadas bounding boxes: {bboxes}")
        print(f"Confianza bounding boxes:{probs} ")
        print("")
        
    return bboxes.astype(int)

#=================================================================================================================

def mostrar_bboxes_cv2(imagen: Union[PIL.Image.Image, np.ndarray],
                       bboxes: np.ndarray,
                       identidades: list=None,
                       device: str='window') -> None:

    
    
    
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser `np.ndarray`, `PIL.Image`. Recibido {type(imagen)}."
        )
        
    if identidades is not None:
        if len(bboxes) != len(identidades):
            raise Exception(
                '`identidades` debe tener el mismo número de elementos que `bboxes`.'
            )
    else:
        identidades = [None] * len(bboxes)

    # Mostrar la imagen y superponer bounding boxes
    # --------------------------------------------------------------------------      
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen).astype(np.float32) / 255
    
    if len(bboxes) > 0:
        
        for i, bbox in enumerate(bboxes):
            
            if identidades[i] is not None:
                cv2.rectangle(
                    img       = imagen,
                    pt1       = (bbox[0], bbox[1]),
                    pt2       = (bbox[2], bbox[3]),
                    color     = (0, 255, 0),
                    thickness = 2
                )
                
                cv2.putText(
                    img       = imagen, 
                    text      = identidades[i], 
                    org       = (bbox[0], bbox[1]-10), 
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1e-3 * imagen.shape[0],
                    color     = (0,255,0),
                    thickness = 2
                )
            else:
                cv2.rectangle(
                    img       = imagen,
                    pt1       = (bbox[0], bbox[1]),
                    pt2       = (bbox[2], bbox[3]),
                    color     = (255, 0, 0),
                    thickness = 2
                )
        
    if device is None:
        return imagen

#=================================================================================================================

def extraer_caras(imagen: Union[PIL.Image.Image, np.ndarray],
                  bboxes: np.ndarray,
                  output_img_size: Union[list, tuple, np.ndarray]=[160, 160]) -> None:
    

    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser np.ndarray, PIL.Image. Recibido {type(imagen)}."
        )
        
    # Recorte de cara
    # --------------------------------------------------------------------------
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen)
        
    if len(bboxes) > 0:
        caras = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cara = imagen[y1:y2, x1:x2]
            # Redimensionamiento del recorte
            cara = Image.fromarray(cara)
            cara = cara.resize(tuple(output_img_size))
            cara = np.array(cara)
            caras.append(cara)
            
    caras = np.stack(caras, axis=0)

    return caras

#=================================================================================================================

def calcular_embeddings(img_caras: np.ndarray, encoder=None,
                        device: str=None) -> np.ndarray: 
    

    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(img_caras, np.ndarray):
        raise Exception(
            f"`img_caras` debe ser np.ndarray {type(img_caras)}."
        )
        
    if img_caras.ndim != 4:
        raise Exception(
            f"`img_caras` debe ser np.ndarray con dimensiones [nº caras, ancho, alto, 3]."
            f" Recibido {img_caras.ndim}."
        )

        
    # Calculo de embedings
    # --------------------------------------------------------------------------
  
    caras = np.moveaxis(img_caras, -1, 1)
    caras = caras.astype(np.float32) / 255
    
    caras = torch.tensor(caras,device=device)      
    embeddings = encoder.forward(caras).detach().cpu().numpy() 

    return embeddings

#=================================================================================================================

def identificar_caras(embeddings: np.ndarray,
                      dic_referencias: dict,
                      threshold_similaridad: float = 0.6) -> list:
        
    identidades = []
        
    for i in range(embeddings.shape[0]):
       
        # Se calcula la similitud con cada uno de los perfiles de referencia.
        similitudes = {}
        for key, value in dic_referencias.items():         
            
            similitudes[key] = 1 - cosine(embeddings[i], np.squeeze(value))
                
        # Se identifica la persona de mayor similitud.
        identidad = max(similitudes, key=similitudes.get)
        # Si la similitud < threshold_similaridad, se etiqueta como None
        if similitudes[identidad] < threshold_similaridad:
            identidad = 'Desconocido'
            
        identidades.append(identidad)
        
    return identidades

#=================================================================================================================

def crear_diccionario_referencias(folder_path:str,
                                  dic_referencia:dict=None,
                                  detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                                  min_face_size: int=160,
                                  thresholds: list=[0.6, 0.7, 0.7],
                                  min_confidence: float=0.9,
                                  encoder=None,
                                  device: str=None,
                                  verbose: bool=True)-> dict:   
    
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not os.path.isdir(folder_path):
        raise Exception(
            f"Directorio {folder_path} no existe."
        )
        
    if len(os.listdir(folder_path) ) == 0:
        raise Exception(
            f"Directorio {folder_path} está vacío."
        )
     
    
    new_dic_referencia = {}
    folders = glob.glob(folder_path + "/*")
    
    for folder in folders:
        
        if platform.system() in ['Linux', 'Darwin']:
            identidad = folder.split("/")[-1]
        else:
            identidad = folder.split("\\")[-1]

        embeddings = []
        # Se lista todas las imagenes .jpg .jpeg .tif .png
        path_imagenes = glob.glob(folder + "/*.jpg")
        path_imagenes.extend(glob.glob(folder + "/*.jpeg"))
        path_imagenes.extend(glob.glob(folder + "/*.tif"))
        path_imagenes.extend(glob.glob(folder + "/*.png"))
        logging.info(f'Total imagenes referencia: {len(path_imagenes)}')
        
        for path_imagen in path_imagenes:
            imagen = Image.open(path_imagen)
            # Si la imagen es RGBA se pasa a RGB
            if np.array(imagen).shape[2] == 4:
                imagen  = np.array(imagen)[:, :, :3]
                imagen  = Image.fromarray(imagen)
                
            bbox = detectar_caras(
                        imagen,
                        detector       = detector,
                        min_confidence = min_confidence,
                        verbose        = False
                    )
            
            if len(bbox) > 1:
                logging.warning(
                    f'Más de 2 caras detectadas en la imagen: {path_imagen}. '
                    f'Se descarta la imagen del diccionario de referencia.'
                )
                continue
                
            if len(bbox) == 0:
                logging.warning(
                    f'No se han detectado caras en la imagen: {path_imagen}.'
                )
                continue
                
            cara = extraer_caras(imagen, bbox)
            embedding = calcular_embeddings(cara, encoder=encoder,device=device)
            embeddings.append(embedding)
        
        if verbose:
            print(f"Identidad: {identidad} --- Imágenes referencia: {len(embeddings)}")
            
  
        embedding_promedio = np.array(embeddings).mean(axis = 0)
        new_dic_referencia[identidad] = embedding_promedio
        
    if dic_referencia is not None:
        dic_referencia.update(new_dic_referencia)
        return dic_referencia
    else:
        return new_dic_referencia    

#=================================================================================================================

def escrituraDB(BASE_DATOS,nombre,dia,mes,año,hora,minutos,segundos,fdb):
    
    BASE_DATOS = BASE_DATOS.append({'NOMBRE':nombre,
                                    'DIA':dia,
                                    'MES':mes,
                                    'AÑO':año,
                                    'HORA':hora,
                                    'MINUTOS':minutos,
                                    'SEGUNDOS':segundos,                                  
                                    
                                    },ignore_index=True)
    BASE_DATOS = BASE_DATOS[['NOMBRE', 'DIA', 'MES', 'AÑO', 'HORA', 'MINUTOS', 'SEGUNDOS']]
    try:
      BASE_DATOS.to_csv(fdb, index=False)  
    except:     
      print("Base datos CSV no guardada")   
    return BASE_DATOS

#=================================================================================================================

def ahora():
    now =  datetime.now()
    tiempo = now.strftime("%Y-%m-%d %H:%M:%S:%f")
    
    return tiempo

#=================================================================================================================

def pipeline_deteccion_webcam(dic_referencia: dict,
                             output_device: str = 'window',
                             path_output_video: str=os.getcwd(),
                             detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                             keep_all: bool=True,
                             min_face_size: int=160,
                             thresholds: list=[0.6, 0.7, 0.7],
                             device: str=None,
                             min_confidence: float=0.9,
                             fix_bbox: bool=True,
                             output_img_size: Union[list, tuple, np.ndarray]=[160, 160],
                             encoder=None,
                             threshold_similaridad: float=0.6,
                             ax=None,
                             verbose=False,
                             band:bool=True,
                             BASE_DATOS=dict)-> None:    
    width = 720
    heigth = 720
    dsize = (width,heigth)
    
    # Variable que controla el encendido y apagado de los led del espejo.
    led_espejo = 0
    
    
    # Ajustes de la cámara para que sea capas de ver a las personas.
    requests.get('http://' + IP_esp32 + '/control?var=framesize&val=11')        # Resolución
    requests.get('http://' + IP_esp32 + '/control?var=brightness&val=1')        # Brillo de imagen
    requests.get('http://' + IP_esp32 + '/control?var=contrast&val=0')         # Contraste
    requests.get('http://' + IP_esp32 + '/control?var=gainceiling&val=2')       # Techo de Ganancia
    requests.get('http://' + IP_esp32 + '/control?var=wb_mode&val=0')           # Modo de los blanco y negros.
    requests.get('http://' + IP_esp32 + '/control?var=led_intensity&val=0')    # Intensidad del led Flash
    
    
    # Dirección para la captura de la imagen de la persona.
    url_capture = 'http://' + IP_esp32 + '/capture?_cb'
    
    capture = cv2.VideoCapture(url_capture)
    
    
    # Dimensiones de la zona de interes.
    xi = 650
    xf = 1200
    yi = 10
    yf = 552
    gamma = 0.5
    
    while True:
        
     capture.open(url_capture) # Abrimos la url.
   
     ret, frame = capture.read() # Capturamos el fame

     if ret != False:    

        frame_cut = frame[yi:yf,xi:xf]
        
        ancho = frame_cut.shape[1] #columnas
        alto = frame_cut.shape[0] # filas
        
        M = cv2.getRotationMatrix2D((ancho//2,alto//2),270,1)
        frame_cut = cv2.warpAffine(frame_cut,M,(ancho,alto))
        
        gamma_corrected = np.array(255*(frame_cut / 255) ** gamma, dtype = 'uint8')
        frame_cut = gamma_corrected
                
        bboxes = detectar_caras(
                        imagen         = frame_cut,
                        detector       = detector,
                        keep_all       = keep_all,
                        min_face_size  = min_face_size,
                        thresholds     = thresholds,
                        device         = device,
                        min_confidence = min_confidence,
                        fix_bbox       = fix_bbox
                      )
        
        if len(bboxes) == 0:
            logging.info('No se han detectado caras en la imagen.')   
            
            led_espejo = led_DB(identidad= 'off',ledEspejo=led_espejo)
                             
        else:
            
            caras = extraer_caras(
                        imagen = frame_cut,
                        bboxes = bboxes
                    )
            
            embeddings = calcular_embeddings(
                            img_caras = caras,
                            encoder   = encoder,
                            device=device
                         )
            
            identidades = identificar_caras(
                             embeddings     = embeddings,
                             dic_referencias = dic_referencias,
                             threshold_similaridad = threshold_similaridad
                          )
            
            mostrar_bboxes_cv2(
                    imagen      = frame_cut,
                    bboxes      = bboxes,
                    identidades = identidades,
                    device = output_device
                  )
            
            from datetime import datetime    
            now = datetime.now()
            fecha = now.strftime('%d_%m_%Y_%H_%M_%S')
            dia = now.strftime('%d')
            mes = now.strftime('%m')
            año = now.strftime('%Y')
            hora = now.strftime('%H')
            minutos = now.strftime('%M')
            segundos =now.strftime('%S')
            
# ================================================================================================================                
        # Condiciones para los leds.
            
            # print('Empezamos a evaluar las condiciones.')
                        
            s = 0
            
            if identidades[0] == 'Desconocido' and led_espejo == 0:
                led_espejo = led_DB(identidad= identidades[0].lower(), ledEspejo= led_espejo)
            
            elif led_espejo == 1:
                
                if identidades[0] == 'Desconocido' and led_espejo == 1:
                    led_espejo = led_DB(identidad= identidades[0].lower(), ledEspejo= led_espejo)
                    
                    os.chdir('/home/vit/Documentos/Josuep/face_esp32/Programa_final/Data_Vultur/Desconocido')            
                    for h in range(len(caras)):
                     cv2.imwrite('Desconocido_' +'_'+fecha+'_'+str(s)+'.png', caras[h])  
                     s = s+1
                    
                elif identidades[0] != 'Desconocido' and led_espejo == 1:
                    led_espejo = led_DB(identidad= identidades[0].lower(), ledEspejo= led_espejo)
    
                    nombre=identidades[0]
                    print(nombre)
                    os.chdir('/home/vit/Documentos/Josuep/face_esp32/Programa_final/Data_Vultur/Registro')                           
                    BASE_DATOS=escrituraDB(BASE_DATOS,nombre,dia,mes,año,hora,minutos,segundos,fdb)
                    s = s+1

# ================================================================================================================                                         

        if band == True:
            # print("Mostramos la imagen de la cámara.")
             
            frame =cv2.resize(frame_cut,dsize)
             
            cv2.imshow(output_device, frame)
            if cv2.waitKey(1)&0xFF==ord("q"): 
                break 
        
    # print('Destruimos la imagen de la cámara.')
    capture.release()
    cv2.destroyAllWindows()           
    
    
# ================================================================================================================    
# Función de escritura en la base de datos para el encendido y apagado de los leds.

def led_DB(identidad,ledEspejo):
    
    identidad = identidad.lower()
    
    if (identidad == 'off' and ledEspejo == 0):
        # No hay nadie frente de la cámara.
        # No hay razón para encender algún led.
        
        data = {"led_red":"OFF","led_green":"OFF","led_mirrow":"OFF"}
        requests.put(f"{IP_API}/api/entradas/face_recognition/update/1", json = data)
        ledEspejo = 0
        return ledEspejo
    
    else:
        if identidad == 'desconocido' and ledEspejo == 0:
            # Vio una persona pero no sabe quien es debido a la falta de iluminación.
            # Encender los leds del espejo, apagar el led rojo y esperar la siguiente respuesta.
            
            data = {"led_red":"OFF","led_green":"OFF","led_mirrow":"ON"}
            requests.put(f"{IP_API}/api/entradas/face_recognition/update/1", json = data)
            
            ledEspejo = 1
            return ledEspejo
        
        elif identidad == 'desconocido' and ledEspejo == 1:
            # Vio a la persona pero no lo conoce.
            # Apagar los led del espejo y encender led rojo unos 3 segundos para luego apagarlo.
            
            data = {"led_red":"ON","led_green":"OFF","led_mirrow":"OFF"}
            requests.put(f"{IP_API}/api/entradas/face_recognition/update/1", json = data)
            

            time.sleep(1.5)    
            
            data = {"led_red":"OFF","led_green":"OFF","led_mirrow":"OFF"}
            requests.put(f"{IP_API}/api/entradas/face_recognition/update/1", json = data)
            
            ledEspejo = 0
            return ledEspejo
            
        elif identidad != 'desconocido' and ledEspejo == 1:
            # Vio a la persona y la reconocio.
            # Apagar los leds del espejo y encender el led verde por 3 segundos para luego apagarlo.
            
            data = {"led_red":"OFF","led_green":"ON","led_mirrow":"OFF"}
            requests.put(f"{IP_API}/api/entradas/face_recognition/update/1", json = data)

            time.sleep(1.5)        
    
            data = {"led_red":"OFF","led_green":"OFF","led_mirrow":"OFF"}
            requests.put(f"{IP_API}/api/entradas/face_recognition/update/1", json = data)
            
            ledEspejo = 0
            return ledEspejo
            

    
#=================================================================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
device = 'cpu'
           
detector = MTCNN(       post_process  = False,
                        device        = device
                   )

encoder = InceptionResnetV1(
                        pretrained = 'vggface2',
                        classify   = False,
                        device     = device
                   ).eval()

if entrenar:
    dic_referencias = crear_diccionario_referencias(
    folder_path='/home/vit/Documentos/Josuep/face_esp32/Programa_final/Data_Vultur/Base_Datos/',
    min_face_size=320,
    min_confidence=0.9,
    device=device,
    verbose=True,
    detector = detector,
    encoder = encoder
)
    os.chdir('/home/vit/Documentos/Josuep/face_esp32/Programa_final/Data_Vultur/') 
    # # Save
    dictionary = dic_referencias
    np.save('Base_datos.npy', dictionary)     
else:
    os.chdir('/home/vit/Documentos/Josuep/face_esp32/Programa_final/Data_Vultur/') 
    dic_referencias= np.load('Base_datos.npy',allow_pickle='TRUE').item()
    
    tiempo = ahora() 
    tiempoc = tiempo.replace(' ','_') 
    fdb = '/home/vit/Documentos/Josuep/face_esp32/Programa_final/Data_Vultur/Registro/BASE_DATOS_' + tiempoc +".csv"
    BASE_DATOS = pd.DataFrame()

    
    pipeline_deteccion_webcam(
        dic_referencia        = dic_referencias,
        threshold_similaridad = 0.6,
        min_face_size=320,    # Para cambiar tamaño de la imagen debe ir a ~/.local/lib/python3.10/site-packages/facenet_pytorch/models/mtcnn.py
        device=device,
        detector = detector,
        encoder = encoder,
        band=band,
        BASE_DATOS = BASE_DATOS
    )