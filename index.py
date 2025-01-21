import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta
import threading
import uuid

class FaceRecognitionSystem:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.faces_db = {}
        self.mode = "reconocimiento"
        self.cap = None
        self.is_running = True
        self.current_frame = None
        self.current_detections = []
        self.last_welcome_times = {}  # Para almacenar el último tiempo de bienvenida por usuario
        
        # Cargar base de datos si existe
        if os.path.exists('faces_db.pkl'):
            with open('faces_db.pkl', 'rb') as f:
                self.faces_db = pickle.load(f)
        
        self.root = tk.Tk()
        self.root.title("Sistema de Reconocimiento Facial")
        self.root.geometry("400x250")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.setup_gui()

        # Hilo para procesamiento de video
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.start()

    def on_closing(self):
        if messagebox.askokcancel("Salir", "¿Desea cerrar la aplicación?"):
            self.is_running = False
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            self.root.quit()
            self.root.destroy()

    def setup_gui(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        tk.Label(main_frame, text="Sistema de Reconocimiento Facial").pack()
        
        self.name_entry = tk.Entry(main_frame)
        self.name_entry.pack(pady=10)
        
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Registrar", command=self.register_current_faces).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Cerrar", command=self.on_closing).pack(side=tk.LEFT, padx=10)

    def save_database(self):
        with open('faces_db.pkl', 'wb') as f:
            pickle.dump(self.faces_db, f)
        
        with open('faces_database.txt', 'w') as f:
            f.write("Base de datos de rostros registrados\n")
            f.write("====================================\n\n")
            for user_id, data in self.faces_db.items():
                f.write(f"ID: {user_id}\n")
                f.write(f"Nombre: {data['name']}\n")
                f.write(f"Fecha de registro: {data['date']}\n")
                f.write("------------------------------------\n")

    def extract_face_features(self, frame, face_landmarks):
        # Obtener un vector de características a partir de los puntos de la malla facial
        face_vector = []
        for landmark in face_landmarks:
            face_vector.append([landmark.x, landmark.y, landmark.z])
        return np.array(face_vector).flatten()

    def register_face(self, frame, face_landmarks, name):
        features = self.extract_face_features(frame, face_landmarks)
        if features is not None:
            # Generar un identificador único (UUID)
            user_id = str(uuid.uuid4())
            
            # Guardar los datos en la base de datos con el identificador único
            self.faces_db[user_id] = {
                'name': name,
                'features': features,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.save_database()
            return True
        return False

    def recognize_face(self, frame, face_landmarks, threshold=1.0):
        if not self.faces_db:
            return "No hay rostros registrados"
        
        features = self.extract_face_features(frame, face_landmarks)
        if features is None:
            return "Error en detección"
        
        min_dist = float('inf')
        best_match = None
        
        # Comparar las características de este rostro con todos los rostros registrados
        for user_id, data in self.faces_db.items():
            dist = np.linalg.norm(features - data['features'])
            if dist < min_dist:
                min_dist = dist
                best_match = data['name']
        
        # Comparar la distancia con un umbral ajustable
        if min_dist > threshold:  # Si la distancia es mayor que el umbral, no se reconoce como la misma persona
            return "Usuario no identificado"
        return best_match

    def process_video(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara")
            return

        window_name = 'Sistema de Reconocimiento Facial'
        cv2.namedWindow(window_name)
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while self.is_running:
                success, frame = self.cap.read()
                if not success:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                
                self.current_frame = frame.copy()
                self.current_detections = []
                
                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        self.mp_draw.draw_landmarks(frame, landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)
                        
                        # Ahora landmarks es un objeto NormalizedLandmarkList, accedemos a cada punto de referencia correctamente
                        name = self.recognize_face(frame, landmarks.landmark)  # Corregido: usar landmarks.landmark
                        if name == "Usuario no identificado":
                            cv2.putText(frame, name, (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, name, (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # Mostrar ventana flotante con el mensaje de bienvenida, si no se ha mostrado en el último minuto
                            self.show_welcome_window(name)
                        
                        self.current_detections.append(landmarks)
                
                cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def show_welcome_window(self, name):
        # Comprobar si han pasado al menos 1 minuto desde la última vez que se mostró el mensaje
        current_time = datetime.now()
        last_time = self.last_welcome_times.get(name)
        
        if last_time is None or current_time - last_time > timedelta(minutes=1):
            # Si es el primer reconocimiento o han pasado más de 1 minuto, mostrar el mensaje
            welcome_window = tk.Toplevel(self.root)
            welcome_window.title("Bienvenido")
            welcome_window.geometry("300x150")
            welcome_window.resizable(False, False)
            
            welcome_label = tk.Label(welcome_window, text=f"Bienvenido {name}!", font=("Arial", 14))
            welcome_label.pack(expand=True)

            # Cerrar la ventana flotante después de 3 segundos
            welcome_window.after(3000, welcome_window.destroy)

            # Registrar la hora en que se mostró el mensaje
            self.last_welcome_times[name] = current_time

    def register_current_faces(self):
        if self.current_frame is None or len(self.current_detections) == 0:
            messagebox.showerror("Error", "No se detectaron rostros para registrar")
            return

        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Por favor ingrese un nombre")
            return

        # Registrar cada rostro detectado
        registered_count = 0
        for landmarks in self.current_detections:
            if self.register_face(self.current_frame, landmarks.landmark, name):  # Corregido: landmarks.landmark
                registered_count += 1
        
        if registered_count > 0:
            messagebox.showinfo("Éxito", f"{registered_count} rostros registrados exitosamente")
            self.name_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "No se pudo registrar los rostros detectados")

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    system = FaceRecognitionSystem()
    system.run()
