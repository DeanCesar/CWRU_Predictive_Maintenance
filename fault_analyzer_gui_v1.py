# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 06:50:22 2024

@author: PC
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextBrowser, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np
import scipy.io
from sklearn.preprocessing import LabelEncoder
import keras.models as KM
from collections import Counter
import os

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Cargar el modelo al inicializar la interfaz
        self.loadModel()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Bearing ANN Analyzer')
        self.setGeometry(100, 100, 500, 550)

        # Widgets
        self.label_file_name = QLabel('Nombre del archivo: No seleccionado')
        font = QFont("Arial", 20)
        self.label_file_name.setFont(font)
        
        self.label_results = QLabel('Resultados:')
        self.label_results.setFont(font)
        self.text_browser_results = QTextBrowser()
        
        self.button_load_file = QPushButton('Cargar Archivo', self)
        self.button_analyze = QPushButton('Analizar', self)
        
        self.button_analyze.setFont(font)
        self.button_load_file.setFont(font)
        self.text_browser_results.setFont(font)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label_file_name)
        layout.addWidget(self.button_load_file)
        layout.addWidget(self.button_analyze)
        layout.addWidget(self.label_results)
        layout.addWidget(self.text_browser_results)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Conexiones de los botones
        self.button_load_file.clicked.connect(self.loadFile)
        self.button_load_file.setStyleSheet("background:green; color: white")
        self.button_analyze.clicked.connect(self.analyzeFile)
        self.button_analyze.setStyleSheet("background:blue; color: white")

    def loadModel(self):
        # Cargar el modelo una vez al inicio
        MODEL_PATH = 'model.h5'
        self.model = KM.load_model(MODEL_PATH)

        # Crear encoder
        labels = np.array(['14B', '14IR', '14OR', '21B', '21IR', '21OR21', '21OR3', '21OR6', '7B', '7IR', '7OR12',
                           '7OR3', '7OR6', 'N'])
        self.encoder = LabelEncoder()
        self.encoder.fit(labels)

    def loadFile(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Seleccionar archivo .mat', '', 'MAT Files (*.mat)')
        name_path = os.path.basename(file_path)
        
        if file_path:
            self.label_file_name.setText(f'Nombre del archivo: <b>{name_path}</b>')

            # Guardar la ruta del archivo para su posterior uso
            self.file_path = file_path
            self.text_browser_results.setPlainText("")

    def analyzeFile(self):
        try:
            # Cargar y procesar el archivo .mat
            mat = scipy.io.loadmat(self.file_path)
            key_name = list(mat.keys())[3]
            DE_data = mat.get(key_name)
            mat_vector = DE_data.reshape(len(DE_data))

            win_len = 1000
            stride = 200

            X = []
            for i in np.arange(0, len(mat_vector) - (win_len), stride):
                temp = mat_vector[i:i + win_len]
                X.append(temp)

            X_test = np.array(X)

            # Obtener predicciones
            y_viz = self.model.predict(X_test)
            Y_pred = list(inv_Transform_result(y_viz, self.encoder))

            # Calcular porcentajes
            conteo_cadenas = Counter(Y_pred)
            total_elementos = len(Y_pred)
            porcentajes = {cadena: count / total_elementos * 100 for cadena, count in conteo_cadenas.items()}

            # Mostrar resultados en el cuadro de texto
            result_text = ''
            for cadena, porcentaje in sorted(porcentajes.items(), key=lambda x: x[1], reverse=True):
                result_text += f"<b><font color='brown'>{cadena}:</font></b> {porcentaje:.2f}%<br>"
            self.text_browser_results.setHtml(result_text)

        except Exception as e:
            self.text_browser_results.setPlainText(f"Error al analizar el archivo: {str(e)}")



def inv_Transform_result(y_pred, encoder):
    y_pred = y_pred.argmax(axis=1)
    y_pred = encoder.inverse_transform(y_pred)
    return y_pred


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())