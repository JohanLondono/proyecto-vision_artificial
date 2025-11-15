"""
Actividad 18: Captura de Video con Cámara en Python
Universidad del Quindío - Visión Artificial
Implementación completa de captura de video y imágenes
"""

import cv2
import os
import numpy as np

def verificar_camara():
    """Verifica si la cámara está disponible"""
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return False
    camara.release()
    return True

def mostrar_camara_vivo():
    """Muestra la cámara en tiempo real"""
    print("Iniciando vista en vivo de la cámara...")
    print("Presiona 'q' para salir")
    
    camara = cv2.VideoCapture(0)
    
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return
    
    while True:
        ret, frame = camara.read()
        if not ret:
            print("No se pudo leer el cuadro")
            break
        
        # Mostrar la imagen
        cv2.imshow('Camara en vivo - Presiona Q para salir', frame)
        
        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camara.release()
    cv2.destroyAllWindows()
    print("Vista en vivo cerrada")

def capturar_imagenes():
    """Captura imágenes individuales con barra espaciadora"""
    print("Modo captura de imágenes")
    print("Presiona ESPACIO para capturar imagen, 'q' para salir")
    
    camara = cv2.VideoCapture(0)
    
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return
    
    # Crear carpeta de imágenes si no existe
    if not os.path.exists('imagenes_capturadas'):
        os.makedirs('imagenes_capturadas')
    
    contador = 0
    
    while True:
        ret, frame = camara.read()
        if not ret:
            break
        
        cv2.imshow("Captura de imagenes - ESPACIO: capturar, Q: salir", frame)
        
        tecla = cv2.waitKey(1) & 0xFF
        
        if tecla == ord(' '):  # Barra espaciadora
            nombre_imagen = f"imagenes_capturadas/captura_{contador:03d}.jpg"
            cv2.imwrite(nombre_imagen, frame)
            print(f"Imagen guardada como {nombre_imagen}")
            contador += 1
            
        elif tecla == ord('q'):  # Salir con 'q'
            break
    
    camara.release()
    cv2.destroyAllWindows()
    print(f"Total de imágenes capturadas: {contador}")

def grabar_video():
    """Graba video completo"""
    print("Iniciando grabación de video...")
    print("Presiona 'q' para detener la grabación")
    
    camara = cv2.VideoCapture(0)
    
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return
    
    # Obtener dimensiones del frame
    width = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configurar el codec y crear VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Crear carpeta de videos si no existe
    if not os.path.exists('videos_grabados'):
        os.makedirs('videos_grabados')
    
    contador_videos = len([f for f in os.listdir('videos_grabados') if f.endswith('.avi')])
    nombre_video = f'videos_grabados/video_{contador_videos:03d}.avi'
    
    salida = cv2.VideoWriter(nombre_video, fourcc, 20.0, (width, height))
    
    print(f"GRABANDO... Archivo: {nombre_video}")
    
    while True:
        ret, frame = camara.read()
        if not ret:
            break
        
        # Escribir frame al archivo de video
        salida.write(frame)
        
        # Mostrar el frame con indicador de grabación
        cv2.putText(frame, "GRABANDO - Presiona Q para detener", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Grabando video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camara.release()
    salida.release()
    cv2.destroyAllWindows()
    print(f"Video guardado como {nombre_video}")

def capturar_tres_imagenes_consecutivas():
    """Ejercicio: Captura tres imágenes consecutivas"""
    print("Capturando 3 imágenes consecutivas...")
    print("Presiona ESPACIO para iniciar la secuencia")
    
    camara = cv2.VideoCapture(0)
    
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return
    
    # Crear carpeta si no existe
    if not os.path.exists('imagenes_ejercicios'):
        os.makedirs('imagenes_ejercicios')
    
    while True:
        ret, frame = camara.read()
        if not ret:
            break
        
        cv2.imshow("Presiona ESPACIO para iniciar secuencia de 3 fotos", frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Capturar 3 imágenes consecutivas
            for i in range(3):
                ret, frame = camara.read()
                if ret:
                    nombre = f"imagenes_ejercicios/secuencia_{i+1}.jpg"
                    cv2.imwrite(nombre, frame)
                    print(f"Imagen {i+1}/3 guardada: {nombre}")
                    cv2.imshow(f"Capturada {i+1}/3", frame)
                    cv2.waitKey(1000)  # Esperar 1 segundo entre capturas
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camara.release()
    cv2.destroyAllWindows()
    print("Secuencia de 3 imágenes completada")

def grabar_video_10_segundos():
    """Ejercicio: Graba 10 segundos de video"""
    print("Grabando video de 10 segundos...")
    
    camara = cv2.VideoCapture(0)
    
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return
    
    width = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    if not os.path.exists('videos_ejercicios'):
        os.makedirs('videos_ejercicios')
    
    salida = cv2.VideoWriter('videos_ejercicios/video_10_segundos.avi', fourcc, 20.0, (width, height))
    
    import time
    inicio = time.time()
    
    print("Grabando por 10 segundos...")
    
    while True:
        ret, frame = camara.read()
        if not ret:
            break
        
        salida.write(frame)
        
        tiempo_transcurrido = time.time() - inicio
        tiempo_restante = 10 - tiempo_transcurrido
        
        if tiempo_restante <= 0:
            break
        
        # Mostrar contador
        cv2.putText(frame, f"Tiempo restante: {tiempo_restante:.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Grabando 10 segundos', frame)
        
        cv2.waitKey(1)
    
    camara.release()
    salida.release()
    cv2.destroyAllWindows()
    print("Video de 10 segundos guardado en videos_ejercicios/video_10_segundos.avi")

def detector_color_rojo():
    """Ejercicio: Detector que captura imagen si detecta mucho color rojo"""
    print("Detector de color rojo activado")
    print("Se mostrarán dos ventanas en tiempo real:")
    print("- Video Original (izquierda)")
    print("- Detección de Rojo en tiempo real (derecha)")
    print("El sistema capturará automáticamente cuando detecte suficiente color rojo")
    print("Presiona 'q' para salir")
    
    camara = cv2.VideoCapture(0)
    
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return
    
    if not os.path.exists('detecciones_rojo'):
        os.makedirs('detecciones_rojo')
    
    contador = 0
    umbral_rojo = 5.0  # Porcentaje mínimo de píxeles rojos
    
    # Posicionar las ventanas para que no se superpongan
    cv2.namedWindow('Video Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Deteccion de Rojo - Tiempo Real', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Video Original', 100, 100)
    cv2.moveWindow('Deteccion de Rojo - Tiempo Real', 750, 100)
    
    print(f"Analizando color rojo en tiempo real... (Umbral: {umbral_rojo}%)")
    print("Teclas: 'q'=salir, 'r'=reset contador, 's'=captura manual")
    
    while True:
        ret, frame = camara.read()
        if not ret:
            break
        
        # Convertir a HSV para mejor detección de color rojo
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Definir rangos para color rojo en HSV
        # Rango 1: Rojos en el extremo inferior del espectro (0-10)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        # Rango 2: Rojos en el extremo superior del espectro (170-180)
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Crear máscaras para ambos rangos de rojo
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Calcular porcentaje de píxeles rojos
        pixels_rojos = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        porcentaje_rojo = (pixels_rojos / total_pixels) * 100
        
        # Crear copia del frame original para mostrar información
        frame_info = frame.copy()
        
        # Mostrar información en el frame original
        cv2.putText(frame_info, f"Rojo: {porcentaje_rojo:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_info, f"Umbral: {umbral_rojo}%", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_info, f"Capturas: {contador}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Añadir información en la máscara de detección
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convertir a 3 canales para texto en color
        cv2.putText(mask_display, f"Pixeles rojos: {pixels_rojos}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(mask_display, f"Porcentaje: {porcentaje_rojo:.1f}%", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar estado de detección
        if porcentaje_rojo > umbral_rojo:
            cv2.putText(frame_info, "ROJO DETECTADO!", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(mask_display, "DETECTADO!", 
                       (10, mask_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Solo capturar una imagen cuando se detecta por primera vez (no en cada frame)
            # Para evitar capturar múltiples imágenes del mismo evento
            static_vars = getattr(detector_color_rojo, 'static_vars', {'last_detection': False})
            if not static_vars['last_detection']:  # Solo capturar cuando cambia de no detectado a detectado
                nombre = f"detecciones_rojo/rojo_detectado_{contador:03d}.jpg"
                cv2.imwrite(nombre, frame)
                
                # También guardar la máscara
                nombre_mask = f"detecciones_rojo/mascara_rojo_{contador:03d}.jpg"
                cv2.imwrite(nombre_mask, mask)
                
                contador += 1
            
            static_vars['last_detection'] = True
        else:
            static_vars = getattr(detector_color_rojo, 'static_vars', {'last_detection': False})
            static_vars['last_detection'] = False
        
        detector_color_rojo.static_vars = static_vars
        
        # Mostrar ambas ventanas en tiempo real
        cv2.imshow('Video Original', frame_info)
        cv2.imshow('Deteccion de Rojo - Tiempo Real', mask_display)
        
        # Verificar tecla presionada
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset contador con 'r'
            contador = 0
            print("Contador reiniciado")
        elif key == ord('s'):  # Captura manual con 's'
            nombre_manual = f"detecciones_rojo/captura_manual_{contador:03d}.jpg"
            cv2.imwrite(nombre_manual, frame)
            nombre_mask_manual = f"detecciones_rojo/mascara_manual_{contador:03d}.jpg"
            cv2.imwrite(nombre_mask_manual, mask)
            print(f"Captura manual guardada: {nombre_manual}")
            contador += 1
    
    camara.release()
    cv2.destroyAllWindows()
    print(f"Detección finalizada. Total de capturas: {contador}")
    print("Archivos guardados en la carpeta 'detecciones_rojo/'")

def crear_mosaico():
    """Ejercicio: Crear mosaico con cuatro capturas diferentes"""
    print("Creando mosaico con 4 capturas...")
    print("Presiona ESPACIO 4 veces para capturar las imágenes del mosaico")
    
    camara = cv2.VideoCapture(0)
    
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return
    
    imagenes = []
    contador = 0
    
    while contador < 4:
        ret, frame = camara.read()
        if not ret:
            break
        
        cv2.putText(frame, f"Captura {contador + 1}/4 - Presiona ESPACIO", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Capturando para mosaico', frame)
        
        tecla = cv2.waitKey(1) & 0xFF
        
        if tecla == ord(' '):
            # Redimensionar imagen para el mosaico
            frame_small = cv2.resize(frame, (320, 240))
            imagenes.append(frame_small)
            print(f"Imagen {contador + 1}/4 capturada")
            contador += 1
            cv2.waitKey(500)  # Pequeña pausa
        elif tecla == ord('q'):
            break
    
    if len(imagenes) == 4:
        # Crear mosaico 2x2
        fila_superior = np.hstack((imagenes[0], imagenes[1]))
        fila_inferior = np.hstack((imagenes[2], imagenes[3]))
        mosaico = np.vstack((fila_superior, fila_inferior))
        
        # Guardar mosaico
        if not os.path.exists('mosaicos'):
            os.makedirs('mosaicos')
        
        cv2.imwrite('mosaicos/mosaico_4_capturas.jpg', mosaico)
        
        # Mostrar resultado
        cv2.imshow('Mosaico 2x2', mosaico)
        print("Mosaico creado y guardado en mosaicos/mosaico_4_capturas.jpg")
        print("Presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
    
    camara.release()
    cv2.destroyAllWindows()

def comparar_iluminacion():
    """Ejercicio: Comparar dos imágenes con diferentes condiciones de iluminación"""
    print("Comparación de imágenes con diferentes iluminaciones")
    print("Se capturarán 2 imágenes para comparar")
    
    camara = cv2.VideoCapture(0)
    
    if not camara.isOpened():
        print("No se pudo acceder a la cámara")
        return
    
    imagenes = []
    nombres = ["primera_iluminacion", "segunda_iluminacion"]
    
    for i in range(2):
        print(f"\nPreparándose para captura {i+1}/2")
        print(f"Ajusta la iluminación y presiona ESPACIO para capturar")
        
        while True:
            ret, frame = camara.read()
            if not ret:
                break
            
            cv2.putText(frame, f"Captura {i+1}/2 - ESPACIO para capturar", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Comparacion de iluminacion', frame)
            
            tecla = cv2.waitKey(1) & 0xFF
            
            if tecla == ord(' '):
                imagenes.append(frame.copy())
                
                # Guardar imagen
                if not os.path.exists('comparaciones'):
                    os.makedirs('comparaciones')
                
                cv2.imwrite(f'comparaciones/{nombres[i]}.jpg', frame)
                print(f"Imagen {i+1} capturada y guardada")
                break
            elif tecla == ord('q'):
                camara.release()
                cv2.destroyAllWindows()
                return
    
    camara.release()
    
    if len(imagenes) == 2:
        # Análisis de iluminación
        brillo1 = np.mean(cv2.cvtColor(imagenes[0], cv2.COLOR_BGR2GRAY))
        brillo2 = np.mean(cv2.cvtColor(imagenes[1], cv2.COLOR_BGR2GRAY))
        
        # Crear imagen comparativa
        img1_small = cv2.resize(imagenes[0], (320, 240))
        img2_small = cv2.resize(imagenes[1], (320, 240))
        comparacion = np.hstack((img1_small, img2_small))
        
        # Añadir información de brillo
        cv2.putText(comparacion, f"Brillo 1: {brillo1:.1f}", (10, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(comparacion, f"Brillo 2: {brillo2:.1f}", (330, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite('comparaciones/comparacion_iluminacion.jpg', comparacion)
        
        # Mostrar resultados
        cv2.imshow('Comparacion de Iluminacion', comparacion)
        
        print(f"\nAnálisis de iluminación:")
        print(f"Imagen 1 - Brillo promedio: {brillo1:.2f}")
        print(f"Imagen 2 - Brillo promedio: {brillo2:.2f}")
        print(f"Diferencia: {abs(brillo1 - brillo2):.2f}")
        
        if brillo1 > brillo2:
            print("La primera imagen tiene mejor iluminación")
        elif brillo2 > brillo1:
            print("La segunda imagen tiene mejor iluminación")
        else:
            print("Ambas imágenes tienen iluminación similar")
        
        print("\nPresiona cualquier tecla para continuar...")
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def menu_principal():
    """Menú principal de la aplicación"""
    while True:
        print("\n" + "="*60)
        print("CAPTURA DE VIDEO - VISIÓN ARTIFICIAL")
        print("Universidad del Quindío - Actividad 18")
        print("="*60)
        print("1. Vista en vivo de la cámara")
        print("2. Capturar imágenes (con barra espaciadora)")
        print("3. Grabar video")
        print("4. Capturar 3 imágenes consecutivas")
        print("5. Grabar video de 10 segundos")
        print("6. Detector de color rojo")
        print("7. Crear mosaico (4 capturas)")
        print("8. Comparar iluminación")
        print("9. Salir")
        print("="*60)
        
        try:
            opcion = input("Selecciona una opción (1-9): ").strip()
            
            if opcion == '1':
                mostrar_camara_vivo()
            elif opcion == '2':
                capturar_imagenes()
            elif opcion == '3':
                grabar_video()
            elif opcion == '4':
                capturar_tres_imagenes_consecutivas()
            elif opcion == '5':
                grabar_video_10_segundos()
            elif opcion == '6':
                detector_color_rojo()
            elif opcion == '7':
                crear_mosaico()
            elif opcion == '8':
                comparar_iluminacion()
            elif opcion == '9':
                print("¡Hasta luego!")
                break
            else:
                print("Opción no válida. Por favor, selecciona una opción del 1 al 9.")
                
        except KeyboardInterrupt:
            print("\nPrograma interrumpido por el usuario. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")

if __name__ == "__main__":
    print("Verificando acceso a la cámara...")
    
    if verificar_camara():
        print("Cámara detectada correctamente")
        menu_principal()
    else:
        print("No se puede acceder a la cámara.")
        print("Verifica que:")
        print("- La cámara esté conectada")
        print("- No esté siendo usada por otra aplicación")
        print("- Los drivers estén instalados correctamente")