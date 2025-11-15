#!/usr/bin/env python3
"""
Test para la nueva funcionalidad de detección en imagen individual.
Versión simplificada sin emojis para evitar problemas de codificación.
"""

import cv2
import numpy as np
import os
import random


class DetectorImagenIndividual:
    """Versión simplificada del detector para pruebas."""
    
    def __init__(self):
        """Inicializar detector."""
        self.modelos_disponibles = {
            1: {'nombre': 'YOLO-Hat', 'tipo': 'preentrenado', 'descripcion': 'YOLO v8 para detección de sombreros'},
            2: {'nombre': 'Custom-CNN', 'tipo': 'custom', 'descripcion': 'Red neuronal personalizada'},
            3: {'nombre': 'Mask-RCNN', 'tipo': 'segmentacion', 'descripcion': 'Segmentación de sombreros'}
        }
        self.modelo_activo = 1
    
    def _detectar_frame_preentrenado(self, frame, modelo_info):
        """Detecta usando modelos preentrenados con visualización."""
        height, width = frame.shape[:2]
        detecciones = []
        num_detecciones = random.randint(0, 3)
        
        for _ in range(num_detecciones):
            # Generar coordenadas aleatorias realistas
            x = random.randint(50, width - 150)
            y = random.randint(50, height - 150)
            w = random.randint(80, 120)
            h = random.randint(70, 110)
            
            x2 = min(x + w, width)
            y2 = min(y + h, height)
            
            confianza = random.uniform(0.6, 0.9)
            
            deteccion = {
                'bbox': (x, y, x2, y2),
                'confianza': confianza,
                'clase': 'sombrero',
                'modelo': modelo_info['nombre']
            }
            detecciones.append(deteccion)
            
            # Color verde si confianza > 0.7, amarillo si no
            color = (0, 255, 0) if confianza > 0.7 else (0, 255, 255)
            
            # Dibujar rectángulo de detección
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            
            # Etiqueta con confianza
            etiqueta = f"Sombrero {confianza:.2f}"
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(frame, (x, y-th-8), (x+tw+8, y), color, -1)
            cv2.putText(frame, etiqueta, (x+4, y-4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Punto central
            center_x = (x + x2) // 2
            center_y = (y + y2) // 2
            cv2.circle(frame, (center_x, center_y), 4, color, -1)
        
        return len(detecciones)
    
    def detectar_imagen_individual_mejorada(self):
        """Detecta objetos en imagen individual con selección de carpeta e imagen."""
        print("\n" + "="*60)
        print("DETECCION EN IMAGEN INDIVIDUAL - VERSION MEJORADA")
        print("="*60)
        
        try:
            # Mostrar carpetas disponibles como referencia
            print("\nCarpetas disponibles en el directorio actual:")
            try:
                carpetas = [d for d in os.listdir('.') if os.path.isdir(d)]
                for idx, carpeta in enumerate(carpetas[:10], 1):
                    print(f"  {idx}. {carpeta}")
                if len(carpetas) > 10:
                    print(f"  ... y {len(carpetas) - 10} carpetas más")
            except:
                print("  (No se pudieron listar las carpetas)")
            
            # Solicitar ruta de carpeta manualmente
            print(f"\nDirectorio actual: {os.getcwd()}")
            print("Opciones de entrada:")
            print("1. Nombre de carpeta (ej: images)")
            print("2. Ruta relativa (ej: ../otras_imagenes)")
            print("3. Ruta absoluta completa")
            print("0. Cancelar y volver")
            
            entrada_carpeta = input("\nIngrese la ruta de la carpeta de imagenes: ").strip()
            
            if entrada_carpeta == "0" or not entrada_carpeta:
                print("CANCELADO: Operacion cancelada")
                return
            
            # Procesar la entrada
            if not os.path.isabs(entrada_carpeta):
                # Si es ruta relativa, combinar con directorio actual
                carpeta_imagenes = os.path.abspath(entrada_carpeta)
            else:
                carpeta_imagenes = entrada_carpeta
            
            # Verificar que la carpeta existe
            if not os.path.exists(carpeta_imagenes):
                print(f"ERROR: La carpeta '{carpeta_imagenes}' no existe")
                print("Verifique que la ruta sea correcta")
                return
            
            if not os.path.isdir(carpeta_imagenes):
                print(f"ERROR: '{carpeta_imagenes}' no es una carpeta")
                return
                
            print(f"Carpeta seleccionada: {carpeta_imagenes}")
            
            # Buscar imágenes en la carpeta
            extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.avif')
            imagenes_encontradas = []
            
            try:
                for archivo in os.listdir(carpeta_imagenes):
                    if archivo.lower().endswith(extensiones_validas):
                        imagenes_encontradas.append(archivo)
            except PermissionError:
                print(f"ERROR: Sin permisos para acceder a la carpeta '{carpeta_imagenes}'")
                return
            except Exception as e:
                print(f"ERROR al listar archivos: {e}")
                return
            
            if not imagenes_encontradas:
                print(f"ERROR: No se encontraron imagenes en {carpeta_imagenes}")
                print(f"Formatos soportados: {', '.join(extensiones_validas)}")
                return
            
            # Mostrar lista de imágenes disponibles
            print(f"\nImagenes encontradas ({len(imagenes_encontradas)}):")
            print("-" * 50)
            
            for idx, imagen in enumerate(imagenes_encontradas, 1):
                try:
                    ruta_completa = os.path.join(carpeta_imagenes, imagen)
                    tamano = os.path.getsize(ruta_completa)
                    tamano_mb = tamano / (1024 * 1024)
                    print(f"{idx:2d}. {imagen} ({tamano_mb:.2f} MB)")
                except Exception:
                    print(f"{idx:2d}. {imagen}")
            
            # Seleccionar imagen
            print("\n0. Volver al menu principal")
            try:
                seleccion = int(input(f"\nSeleccione imagen (1-{len(imagenes_encontradas)}): "))
                
                if seleccion == 0:
                    print("Volviendo al menu principal...")
                    return
                    
                if seleccion < 1 or seleccion > len(imagenes_encontradas):
                    print("ERROR: Seleccion invalida")
                    return
                    
                imagen_seleccionada = imagenes_encontradas[seleccion - 1]
                ruta_imagen = os.path.join(carpeta_imagenes, imagen_seleccionada)
                
            except ValueError:
                print("ERROR: Ingrese un numero valido")
                return
            
            print(f"\nProcesando: {imagen_seleccionada}")
            print("-" * 50)
            
            # Cargar y procesar imagen
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"ERROR: No se pudo cargar la imagen {imagen_seleccionada}")
                print("Verifique que el archivo no este corrupto")
                return
            
            print(f"Imagen cargada: {imagen.shape[1]}x{imagen.shape[0]} pixeles")
            
            # Obtener información del modelo activo
            modelo_info = self.modelos_disponibles[self.modelo_activo]
            print(f"Usando modelo: {modelo_info['descripcion']}")
            print(f"Tipo: {modelo_info['tipo']}")
            
            # Simular procesamiento
            print("\nProcesando imagen...")
            
            # Aplicar detección
            imagen_procesada = imagen.copy()
            detecciones = self._detectar_frame_preentrenado(imagen_procesada, modelo_info)
            
            # Mostrar resultados
            print(f"\nRESULTADOS DE DETECCION:")
            print(f"- Objetos detectados: {detecciones}")
            print(f"- Confianza promedio: {random.uniform(0.75, 0.95):.3f}")
            print(f"- Tiempo de procesamiento: {random.uniform(0.5, 2.0):.2f}s")
            
            # Mostrar imagen procesada
            ventana_nombre = f"Deteccion - {imagen_seleccionada}"
            cv2.imshow(ventana_nombre, imagen_procesada)
            
            print(f"\nImagen mostrada en ventana: '{ventana_nombre}'")
            print("Controles:")
            print("- Presione cualquier tecla para cerrar la imagen")
            print("- Presione 's' para guardar resultado")
            print("- Presione ESC para cerrar sin guardar")
            
            # Esperar entrada del usuario
            while True:
                tecla = cv2.waitKey(0) & 0xFF
                
                if tecla == ord('s'):
                    # Guardar resultado
                    nombre_base = os.path.splitext(imagen_seleccionada)[0]
                    nombre_salida = f"deteccion_{nombre_base}_resultado.jpg"
                    ruta_salida = os.path.join(carpeta_imagenes, nombre_salida)
                    
                    if cv2.imwrite(ruta_salida, imagen_procesada):
                        print(f"Resultado guardado: {ruta_salida}")
                    else:
                        print("ERROR: No se pudo guardar el archivo")
                    break
                elif tecla == 27:  # ESC
                    print("Cerrando sin guardar...")
                    break
                else:
                    print("Cerrando imagen...")
                    break
            
            cv2.destroyAllWindows()
            print("\nDeteccion completada exitosamente!")
            
        except KeyboardInterrupt:
            print("\nOperacion cancelada por el usuario")
        except Exception as e:
            print(f"ERROR en deteccion: {str(e)}")
            try:
                cv2.destroyAllWindows()
            except:
                pass


def main():
    """Función principal de prueba."""
    detector = DetectorImagenIndividual()
    detector.detectar_imagen_individual_mejorada()


if __name__ == "__main__":
    main()