"""
🧠 C. elegans Neural Analysis Desktop App - Version 2.0
Aplicación de Desktop Mejorada con Visualizaciones Integradas

Esta versión incluye:
- Visualizaciones plotly integradas en la aplicación
- Panel de controles completo
- Análisis PCA y derivadas
- Filtrado por comportamientos
- Múltiples opciones de preprocesamiento
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from scipy.stats import zscore
import os
import tempfile
import webbrowser
import threading
import time

class NeuralAnalysisGUI:
    """Aplicación GUI para análisis neural interactivo"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 C. elegans Neural Analysis - Interactive Desktop App v2.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f5f5f5')
        
        # Variables de datos
        self.data = None
        self.neural_data = None
        self.time_points = None
        self.preprocessed_data = None
        self.pca_results = None
        self.derivatives = None
        self.behavioral_data = None
        
        # Variables de configuración
        self.init_variables()
        
        # Crear interfaz
        self.create_interface()
        
        # Cargar datos por defecto
        self.load_default_data()
        
    def init_variables(self):
        """Inicializar variables de configuración"""
        
        # Visualización
        self.plot_type = tk.StringVar(value="PCA_3D")
        self.preprocessing_method = tk.StringVar(value="StandardScaler")
        self.behavior_filter = tk.StringVar(value="All")
        self.show_derivatives = tk.BooleanVar(value=False)
        self.color_scheme = tk.StringVar(value="viridis")
        
        # PCA
        self.pc_x = tk.IntVar(value=1)
        self.pc_y = tk.IntVar(value=2)
        self.pc_z = tk.IntVar(value=3)
        self.n_components = tk.IntVar(value=10)
        
        # Estilo
        self.show_trajectory = tk.BooleanVar(value=True)
        self.show_points = tk.BooleanVar(value=True)
        self.point_size = tk.IntVar(value=4)
        
        # Preprocesamiento
        self.apply_filter = tk.BooleanVar(value=False)
        self.filter_window = tk.IntVar(value=5)
        
    def create_interface(self):
        """Crear la interfaz principal"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Controles
        self.create_control_panel(main_frame)
        
        # Panel derecho - Visualización
        self.create_visualization_panel(main_frame)
        
        # Barra de estado
        self.create_status_bar()
        
    def create_control_panel(self, parent):
        """Panel de controles a la izquierda"""
        
        # Frame con scroll
        control_frame = ttk.Frame(parent, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Canvas para scroll
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Título
        title = ttk.Label(scrollable_frame, text="🎛️ CONTROLES", 
                         font=('Arial', 12, 'bold'))
        title.pack(pady=10)
        
        # Sección: Datos
        self.create_data_controls(scrollable_frame)
        
        # Sección: Preprocesamiento  
        self.create_preprocessing_controls(scrollable_frame)
        
        # Sección: Visualización
        self.create_visualization_controls(scrollable_frame)
        
        # Sección: PCA
        self.create_pca_controls(scrollable_frame)
        
        # Sección: Comportamientos
        self.create_behavior_controls(scrollable_frame)
        
        # Sección: Estilo
        self.create_style_controls(scrollable_frame)
        
        # Sección: Acciones
        self.create_action_controls(scrollable_frame)
        
    def create_data_controls(self, parent):
        """Controles de datos"""
        
        frame = ttk.LabelFrame(parent, text="📊 Datos", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Cargar archivo
        ttk.Button(frame, text="📁 Cargar CSV", 
                  command=self.load_csv_file).pack(fill=tk.X, pady=2)
        
        # Datos por defecto
        ttk.Button(frame, text="🔄 Datos Sintéticos", 
                  command=self.load_default_data).pack(fill=tk.X, pady=2)
        
        # Info
        self.data_info = ttk.Label(frame, text="Sin datos", foreground="gray")
        self.data_info.pack(anchor=tk.W, pady=2)
        
    def create_preprocessing_controls(self, parent):
        """Controles de preprocesamiento"""
        
        frame = ttk.LabelFrame(parent, text="⚙️ Preprocesamiento", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Método de escalado
        ttk.Label(frame, text="Escalado:").pack(anchor=tk.W)
        scaling_combo = ttk.Combobox(frame, textvariable=self.preprocessing_method,
                                    values=["StandardScaler", "MinMaxScaler", "RobustScaler", "Z-Score", "None"])
        scaling_combo.pack(fill=tk.X, pady=2)
        scaling_combo.bind('<<ComboboxSelected>>', lambda e: self.update_preprocessing())
        
        # Filtro temporal
        filter_frame = ttk.Frame(frame)
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(filter_frame, text="Filtro Temporal", 
                       variable=self.apply_filter,
                       command=self.update_preprocessing).pack(side=tk.LEFT)
        
        ttk.Label(filter_frame, text="Ventana:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(filter_frame, from_=3, to=21, width=5,
                   textvariable=self.filter_window,
                   command=self.update_preprocessing).pack(side=tk.LEFT)
        
    def create_visualization_controls(self, parent):
        """Controles de visualización"""
        
        frame = ttk.LabelFrame(parent, text="📈 Visualización", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Tipos de gráfico
        plot_options = [
            ("📊 PC vs Tiempo", "pc_time"),
            ("📈 PC vs PC", "pc_2d"),
            ("🎆 PCA 3D", "pca_3d"),
            ("🌊 Derivadas vs Tiempo", "deriv_time"),
            ("🔄 Derivadas 2D", "deriv_2d"),
            ("🎯 Derivadas 3D", "deriv_3d")
        ]
        
        for text, value in plot_options:
            ttk.Radiobutton(frame, text=text, variable=self.plot_type, 
                           value=value, command=self.update_plot).pack(anchor=tk.W)
        
        # Mostrar derivadas
        ttk.Checkbutton(frame, text="🔢 Usar Derivadas", 
                       variable=self.show_derivatives,
                       command=self.update_plot).pack(anchor=tk.W, pady=5)
        
    def create_pca_controls(self, parent):
        """Controles de PCA"""
        
        frame = ttk.LabelFrame(parent, text="🎯 PCA", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Número de componentes
        comp_frame = ttk.Frame(frame)
        comp_frame.pack(fill=tk.X)
        ttk.Label(comp_frame, text="Componentes:").pack(side=tk.LEFT)
        ttk.Spinbox(comp_frame, from_=3, to=50, width=5,
                   textvariable=self.n_components,
                   command=self.update_pca).pack(side=tk.RIGHT)
        
        # Seleccionar PCs
        for pc_name, pc_var in [("PC X:", self.pc_x), ("PC Y:", self.pc_y), ("PC Z:", self.pc_z)]:
            pc_frame = ttk.Frame(frame)
            pc_frame.pack(fill=tk.X, pady=2)
            ttk.Label(pc_frame, text=pc_name).pack(side=tk.LEFT)
            ttk.Spinbox(pc_frame, from_=1, to=20, width=5,
                       textvariable=pc_var,
                       command=self.update_plot).pack(side=tk.RIGHT)
        
    def create_behavior_controls(self, parent):
        """Controles de comportamiento"""
        
        frame = ttk.LabelFrame(parent, text="🎭 Comportamientos", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Filtro de comportamiento
        ttk.Label(frame, text="Filtrar:").pack(anchor=tk.W)
        behavior_combo = ttk.Combobox(frame, textvariable=self.behavior_filter,
                                     values=["All", "Forward", "Reverse", "Pause", "Turn"])
        behavior_combo.pack(fill=tk.X, pady=2)
        behavior_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        
        # Identificar comportamientos
        ttk.Button(frame, text="🔍 Identificar", 
                  command=self.identify_behaviors).pack(fill=tk.X, pady=2)
        
        # Info comportamientos
        self.behavior_info = ttk.Label(frame, text="No identificados", foreground="gray")
        self.behavior_info.pack(anchor=tk.W)
        
    def create_style_controls(self, parent):
        """Controles de estilo"""
        
        frame = ttk.LabelFrame(parent, text="🎨 Estilo", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Esquema de colores
        ttk.Label(frame, text="Colores:").pack(anchor=tk.W)
        color_combo = ttk.Combobox(frame, textvariable=self.color_scheme,
                                  values=["viridis", "plasma", "inferno", "turbo", "rainbow"])
        color_combo.pack(fill=tk.X, pady=2)
        color_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        
        # Opciones de display
        ttk.Checkbutton(frame, text="Mostrar Trayectoria", 
                       variable=self.show_trajectory,
                       command=self.update_plot).pack(anchor=tk.W)
        
        ttk.Checkbutton(frame, text="Mostrar Puntos", 
                       variable=self.show_points,
                       command=self.update_plot).pack(anchor=tk.W)
        
        # Tamaño de puntos
        size_frame = ttk.Frame(frame)
        size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(size_frame, text="Tamaño:").pack(side=tk.LEFT)
        ttk.Spinbox(size_frame, from_=1, to=20, width=5,
                   textvariable=self.point_size,
                   command=self.update_plot).pack(side=tk.RIGHT)
        
    def create_action_controls(self, parent):
        """Controles de acciones"""
        
        frame = ttk.LabelFrame(parent, text="🚀 Acciones", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Botones principales
        ttk.Button(frame, text="🔄 Actualizar", 
                  command=self.update_plot).pack(fill=tk.X, pady=2)
        
        ttk.Button(frame, text="💾 Exportar", 
                  command=self.export_plot).pack(fill=tk.X, pady=2)
        
        ttk.Button(frame, text="🌐 Abrir en Navegador", 
                  command=self.open_in_browser).pack(fill=tk.X, pady=2)
        
        ttk.Button(frame, text="🔄 Resetear", 
                  command=self.reset_controls).pack(fill=tk.X, pady=2)
        
    def create_visualization_panel(self, parent):
        """Panel de visualización a la derecha"""
        
        viz_frame = ttk.Frame(parent)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Título
        title = ttk.Label(viz_frame, text="📊 VISUALIZACIÓN INTERACTIVA", 
                         font=('Arial', 12, 'bold'))
        title.pack(pady=10)
        
        # Info del gráfico actual
        self.plot_info = ttk.Label(viz_frame, text="Selecciona un tipo de gráfico", 
                                  font=('Arial', 10), foreground="gray")
        self.plot_info.pack()
        
        # Frame para mostrar info del plot
        self.viz_info_frame = ttk.Frame(viz_frame, relief=tk.SUNKEN, borderwidth=2)
        self.viz_info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label inicial
        self.viz_label = ttk.Label(self.viz_info_frame, 
                                  text="🧠 Carga datos y selecciona visualización\n\n" +
                                       "Los gráficos interactivos se abrirán\n" +
                                       "en tu navegador web para mejor\n" +
                                       "experiencia de usuario.\n\n" +
                                       "✨ Características:\n" +
                                       "• Rotación 3D interactiva\n" +
                                       "• Zoom dinámico\n" +
                                       "• Información detallada\n" +
                                       "• Exportación de imágenes",
                                  font=('Arial', 11), justify=tk.CENTER)
        self.viz_label.pack(expand=True)
        
    def create_status_bar(self):
        """Barra de estado"""
        
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Listo - Carga datos para comenzar")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        
        # Info de versión
        version_label = ttk.Label(status_frame, text="v2.0 - Neural Analysis", 
                                 relief=tk.SUNKEN)
        version_label.pack(side=tk.RIGHT, padx=2, pady=2)
        
    def update_status(self, message):
        """Actualizar estado"""
        self.status_var.set(message)
        self.root.update()
        
    def load_csv_file(self):
        """Cargar archivo CSV"""
        
        file_path = filedialog.askopenfilename(
            title="Cargar datos neurales",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.update_status("Cargando archivo...")
                self.data = pd.read_csv(file_path)
                self.process_data()
                self.update_status(f"Archivo cargado: {os.path.basename(file_path)}")
                messagebox.showinfo("Éxito", "Datos cargados correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando archivo:\n{str(e)}")
                
    def load_default_data(self):
        """Cargar datos sintéticos por defecto"""
        
        try:
            self.update_status("Generando datos sintéticos...")
            
            # Verificar si existe archivo real
            if os.path.exists('neural_data_dataframe.csv'):
                self.data = pd.read_csv('neural_data_dataframe.csv')
            else:
                # Generar datos sintéticos
                self.generate_synthetic_data()
                
            self.process_data()
            self.update_status("Datos sintéticos cargados")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generando datos:\n{str(e)}")
            
    def generate_synthetic_data(self):
        """Generar datos neurales sintéticos"""
        
        np.random.seed(42)
        n_timepoints = 1000
        n_neurons = 30
        
        time = np.linspace(0, 100, n_timepoints)
        
        # Generar patrones neurales diversos
        neural_data = np.zeros((n_timepoints, n_neurons))
        
        for i in range(n_neurons):
            # Diferentes tipos de patrones
            if i < 10:  # Neuronas oscilatorias
                freq = 0.1 + i * 0.02
                neural_data[:, i] = np.sin(2 * np.pi * freq * time) + 0.3 * np.random.randn(n_timepoints)
            elif i < 20:  # Neuronas con bursts
                burst_times = np.random.choice(n_timepoints, size=20, replace=False)
                for bt in burst_times:
                    start, end = max(0, bt-10), min(n_timepoints, bt+10)
                    neural_data[start:end, i] += 2 * np.exp(-0.5 * ((np.arange(start, end) - bt) / 5)**2)
                neural_data[:, i] += 0.2 * np.random.randn(n_timepoints)
            else:  # Neuronas con drift
                drift = np.cumsum(0.01 * np.random.randn(n_timepoints))
                neural_data[:, i] = drift + 0.5 * np.random.randn(n_timepoints)
        
        # Crear DataFrame
        columns = ['Time_minutes'] + [f'Neuron_{i+1:03d}' for i in range(n_neurons)]
        data_dict = {'Time_minutes': time}
        for i in range(n_neurons):
            data_dict[f'Neuron_{i+1:03d}'] = neural_data[:, i]
            
        self.data = pd.DataFrame(data_dict)
        
    def process_data(self):
        """Procesar datos cargados"""
        
        # Identificar columnas de neuronas
        neuron_cols = [col for col in self.data.columns 
                      if 'neuron' in col.lower() or 'Neuron' in col]
        
        if not neuron_cols:
            raise ValueError("No se encontraron columnas de neuronas")
            
        self.neural_data = self.data[neuron_cols].values
        
        # Tiempo
        time_cols = [col for col in self.data.columns if 'time' in col.lower() or 'Time' in col]
        if time_cols:
            self.time_points = self.data[time_cols[0]].values
        else:
            self.time_points = np.arange(len(self.neural_data))
            
        # Actualizar info
        info_text = f"{self.neural_data.shape[0]} × {self.neural_data.shape[1]} (timepoints × neuronas)"
        self.data_info.config(text=info_text, foreground="black")
        
        # Procesar automáticamente
        self.update_preprocessing()
        
    def update_preprocessing(self):
        """Actualizar preprocesamiento"""
        
        if self.neural_data is None:
            return
            
        try:
            self.update_status("Aplicando preprocesamiento...")
            
            data = self.neural_data.copy()
            
            # Filtro temporal
            if self.apply_filter.get():
                window = self.filter_window.get()
                if window % 2 == 0:
                    window += 1
                for i in range(data.shape[1]):
                    if len(data) > window:
                        data[:, i] = savgol_filter(data[:, i], window, 3)
            
            # Escalado
            method = self.preprocessing_method.get()
            if method == "StandardScaler":
                scaler = StandardScaler()
                self.preprocessed_data = scaler.fit_transform(data)
            elif method == "MinMaxScaler":
                scaler = MinMaxScaler()
                self.preprocessed_data = scaler.fit_transform(data)
            elif method == "RobustScaler":
                scaler = RobustScaler()
                self.preprocessed_data = scaler.fit_transform(data)
            elif method == "Z-Score":
                self.preprocessed_data = zscore(data, axis=0, nan_policy='omit')
            else:
                self.preprocessed_data = data
                
            # Actualizar PCA y derivadas
            self.update_pca()
            self.calculate_derivatives()
            
            self.update_status("Preprocesamiento completado")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en preprocesamiento:\n{str(e)}")
            
    def update_pca(self):
        """Actualizar PCA"""
        
        if self.preprocessed_data is None:
            return
            
        try:
            n_comp = min(self.n_components.get(), 
                        self.preprocessed_data.shape[1], 
                        self.preprocessed_data.shape[0])
            
            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(self.preprocessed_data)
            
            self.pca_results = {
                'scores': scores,
                'components': pca.components_,
                'explained_variance': pca.explained_variance_ratio_,
                'pca': pca
            }
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculando PCA:\n{str(e)}")
            
    def calculate_derivatives(self):
        """Calcular derivadas"""
        
        if self.preprocessed_data is None:
            return
            
        try:
            window = 5
            if len(self.preprocessed_data) > window:
                self.derivatives = np.array([
                    savgol_filter(self.preprocessed_data[:, i], window, 3, deriv=1)
                    for i in range(self.preprocessed_data.shape[1])
                ]).T
            else:
                self.derivatives = np.gradient(self.preprocessed_data, axis=0)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error calculando derivadas:\n{str(e)}")
            
    def identify_behaviors(self):
        """Identificar comportamientos"""
        
        if self.pca_results is None:
            messagebox.showwarning("Advertencia", "Primero carga y procesa datos")
            return
            
        try:
            self.update_status("Identificando comportamientos...")
            
            # Clustering en espacio PCA
            pca_data = self.pca_results['scores'][:, :3]
            kmeans = KMeans(n_clusters=4, random_state=42)
            labels = kmeans.fit_predict(pca_data)
            
            behavior_names = {0: "Forward", 1: "Reverse", 2: "Pause", 3: "Turn"}
            
            self.behavioral_data = {
                'labels': labels,
                'names': behavior_names,
                'centroids': kmeans.cluster_centers_
            }
            
            # Actualizar info
            unique_behaviors = len(np.unique(labels))
            self.behavior_info.config(text=f"{unique_behaviors} tipos identificados", 
                                    foreground="black")
            
            self.update_status("Comportamientos identificados")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error identificando comportamientos:\n{str(e)}")
            
    def update_plot(self):
        """Actualizar visualización"""
        
        if self.preprocessed_data is None or self.pca_results is None:
            return
            
        try:
            self.update_status("Generando visualización...")
            
            plot_type = self.plot_type.get()
            
            if plot_type == "pc_time":
                fig = self.create_pc_time_plot()
            elif plot_type == "pc_2d":
                fig = self.create_pc_2d_plot()
            elif plot_type == "pca_3d":
                fig = self.create_pca_3d_plot()
            elif plot_type == "deriv_time":
                fig = self.create_deriv_time_plot()
            elif plot_type == "deriv_2d":
                fig = self.create_deriv_2d_plot()
            elif plot_type == "deriv_3d":
                fig = self.create_deriv_3d_plot()
            else:
                fig = self.create_pca_3d_plot()  # Default
                
            # Guardar y mostrar
            self.current_figure = fig
            self.display_plot_info(plot_type)
            
            self.update_status("Visualización lista - Usa 'Abrir en Navegador'")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generando visualización:\n{str(e)}")
            
    def create_pc_time_plot(self):
        """PC vs Tiempo"""
        
        pc_idx = self.pc_x.get() - 1
        scores = self.get_data_for_plot()
        
        mask = self.get_behavior_mask()
        x_data = self.time_points[mask]
        y_data = scores[mask, pc_idx]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode='lines+markers' if self.show_points.get() else 'lines',
            name=f'PC{pc_idx+1}',
            line=dict(width=2),
            marker=dict(size=self.point_size.get())
        ))
        
        explained_var = self.pca_results['explained_variance'][pc_idx]
        fig.update_layout(
            title=f'PC{pc_idx+1} vs Tiempo (Varianza: {explained_var:.2%})',
            xaxis_title='Tiempo',
            yaxis_title=f'PC{pc_idx+1}',
            template='plotly_white'
        )
        
        return fig
        
    def create_pc_2d_plot(self):
        """PC vs PC (2D)"""
        
        pc_x_idx = self.pc_x.get() - 1
        pc_y_idx = self.pc_y.get() - 1
        scores = self.get_data_for_plot()
        
        mask = self.get_behavior_mask()
        x_data = scores[mask, pc_x_idx]
        y_data = scores[mask, pc_y_idx]
        
        # Color por tiempo o comportamiento
        if self.behavioral_data is not None and self.behavior_filter.get() == "All":
            color_data = self.behavioral_data['labels'][mask]
        else:
            color_data = self.time_points[mask]
            
        fig = go.Figure()
        
        mode = self.get_plot_mode()
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode=mode,
            marker=dict(
                size=self.point_size.get(),
                color=color_data,
                colorscale=self.color_scheme.get(),
                opacity=0.8
            ),
            line=dict(width=2),
            name='Trayectoria'
        ))
        
        explained_var = self.pca_results['explained_variance']
        fig.update_layout(
            title=f'PC{pc_x_idx+1} vs PC{pc_y_idx+1}',
            xaxis_title=f'PC{pc_x_idx+1} ({explained_var[pc_x_idx]:.2%})',
            yaxis_title=f'PC{pc_y_idx+1} ({explained_var[pc_y_idx]:.2%})',
            template='plotly_white'
        )
        
        return fig
        
    def create_pca_3d_plot(self):
        """PCA 3D"""
        
        pc_x_idx = self.pc_x.get() - 1
        pc_y_idx = self.pc_y.get() - 1
        pc_z_idx = self.pc_z.get() - 1
        scores = self.get_data_for_plot()
        
        mask = self.get_behavior_mask()
        x_data = scores[mask, pc_x_idx]
        y_data = scores[mask, pc_y_idx]
        z_data = scores[mask, pc_z_idx]
        
        # Color
        if self.behavioral_data is not None and self.behavior_filter.get() == "All":
            color_data = self.behavioral_data['labels'][mask]
        else:
            color_data = self.time_points[mask]
            
        fig = go.Figure()
        
        mode = self.get_plot_mode()
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode=mode,
            marker=dict(
                size=self.point_size.get(),
                color=color_data,
                colorscale=self.color_scheme.get(),
                opacity=0.8
            ),
            line=dict(width=4),
            name='Trayectoria 3D'
        ))
        
        explained_var = self.pca_results['explained_variance']
        fig.update_layout(
            title='PCA 3D - Trayectoria Neural',
            scene=dict(
                xaxis_title=f'PC{pc_x_idx+1} ({explained_var[pc_x_idx]:.2%})',
                yaxis_title=f'PC{pc_y_idx+1} ({explained_var[pc_y_idx]:.2%})',
                zaxis_title=f'PC{pc_z_idx+1} ({explained_var[pc_z_idx]:.2%})'
            ),
            template='plotly_white'
        )
        
        return fig
        
    def create_deriv_time_plot(self):
        """Derivadas vs Tiempo"""
        
        if self.derivatives is None:
            return go.Figure()
            
        # PCA de derivadas
        pca_deriv = PCA(n_components=min(10, self.derivatives.shape[1]))
        deriv_scores = pca_deriv.fit_transform(self.derivatives)
        
        pc_idx = self.pc_x.get() - 1
        mask = self.get_behavior_mask()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.time_points[mask],
            y=deriv_scores[mask, pc_idx],
            mode='lines+markers' if self.show_points.get() else 'lines',
            name=f'Derivada PC{pc_idx+1}',
            line=dict(width=2),
            marker=dict(size=self.point_size.get())
        ))
        
        fig.update_layout(
            title=f'Derivadas PC{pc_idx+1} vs Tiempo',
            xaxis_title='Tiempo',
            yaxis_title=f'Derivada PC{pc_idx+1}',
            template='plotly_white'
        )
        
        return fig
        
    def create_deriv_2d_plot(self):
        """Derivadas 2D"""
        
        if self.derivatives is None:
            return go.Figure()
            
        pca_deriv = PCA(n_components=min(10, self.derivatives.shape[1]))
        deriv_scores = pca_deriv.fit_transform(self.derivatives)
        
        pc_x_idx = self.pc_x.get() - 1
        pc_y_idx = self.pc_y.get() - 1
        mask = self.get_behavior_mask()
        
        fig = go.Figure()
        
        mode = self.get_plot_mode()
        fig.add_trace(go.Scatter(
            x=deriv_scores[mask, pc_x_idx],
            y=deriv_scores[mask, pc_y_idx],
            mode=mode,
            marker=dict(
                size=self.point_size.get(),
                color=self.time_points[mask],
                colorscale=self.color_scheme.get(),
                opacity=0.8
            ),
            line=dict(width=2),
            name='Derivadas'
        ))
        
        fig.update_layout(
            title='Derivadas 2D - Espacio de Velocidades',
            xaxis_title=f'Derivada PC{pc_x_idx+1}',
            yaxis_title=f'Derivada PC{pc_y_idx+1}',
            template='plotly_white'
        )
        
        return fig
        
    def create_deriv_3d_plot(self):
        """Derivadas 3D"""
        
        if self.derivatives is None:
            return go.Figure()
            
        pca_deriv = PCA(n_components=min(10, self.derivatives.shape[1]))
        deriv_scores = pca_deriv.fit_transform(self.derivatives)
        
        pc_x_idx = self.pc_x.get() - 1
        pc_y_idx = self.pc_y.get() - 1
        pc_z_idx = self.pc_z.get() - 1
        mask = self.get_behavior_mask()
        
        fig = go.Figure()
        
        mode = self.get_plot_mode()
        fig.add_trace(go.Scatter3d(
            x=deriv_scores[mask, pc_x_idx],
            y=deriv_scores[mask, pc_y_idx],
            z=deriv_scores[mask, pc_z_idx],
            mode=mode,
            marker=dict(
                size=self.point_size.get(),
                color=self.time_points[mask],
                colorscale=self.color_scheme.get(),
                opacity=0.8
            ),
            line=dict(width=4),
            name='Derivadas 3D'
        ))
        
        fig.update_layout(
            title='Derivadas 3D - Espacio de Velocidades',
            scene=dict(
                xaxis_title=f'Derivada PC{pc_x_idx+1}',
                yaxis_title=f'Derivada PC{pc_y_idx+1}',
                zaxis_title=f'Derivada PC{pc_z_idx+1}'
            ),
            template='plotly_white'
        )
        
        return fig
        
    def get_data_for_plot(self):
        """Obtener datos para graficar"""
        if self.show_derivatives.get() and self.derivatives is not None:
            pca_deriv = PCA(n_components=min(10, self.derivatives.shape[1]))
            return pca_deriv.fit_transform(self.derivatives)
        else:
            return self.pca_results['scores']
            
    def get_behavior_mask(self):
        """Máscara de comportamiento"""
        if self.behavior_filter.get() == "All" or self.behavioral_data is None:
            return np.ones(len(self.time_points), dtype=bool)
            
        behavior_map = {"Forward": 0, "Reverse": 1, "Pause": 2, "Turn": 3}
        behavior_idx = behavior_map.get(self.behavior_filter.get(), 0)
        return self.behavioral_data['labels'] == behavior_idx
        
    def get_plot_mode(self):
        """Modo de graficado"""
        if self.show_trajectory.get() and self.show_points.get():
            return 'lines+markers'
        elif self.show_trajectory.get():
            return 'lines'
        else:
            return 'markers'
            
    def display_plot_info(self, plot_type):
        """Mostrar información del gráfico"""
        
        # Limpiar frame
        for widget in self.viz_info_frame.winfo_children():
            widget.destroy()
            
        # Info según tipo
        info_texts = {
            "pc_time": "📊 PC vs Tiempo\n\nMuestra la evolución temporal de\nun componente principal específico.\n\nInteracciones:\n• Hover para detalles\n• Zoom con scroll\n• Pan arrastrando",
            "pc_2d": "📈 PC vs PC (2D)\n\nTrayectoria en espacio de dos\ncomponentes principales.\n\nInteracciones:\n• Colores por tiempo/comportamiento\n• Zoom y pan disponibles\n• Click en leyenda",
            "pca_3d": "🎆 PCA 3D\n\nVisualización tridimensional completa\nde la trayectoria neural.\n\nInteracciones:\n• Rotación 3D con mouse\n• Zoom con scroll\n• Colores dinámicos",
            "deriv_time": "🌊 Derivadas vs Tiempo\n\nVelocidades de cambio neural\na lo largo del tiempo.\n\nInteracciones:\n• Muestra dinámicas temporales\n• Identifica transiciones rápidas",
            "deriv_2d": "🔄 Derivadas 2D\n\nEspacio de velocidades de cambio\nen dos dimensiones.\n\nInteracciones:\n• Patrones de aceleración\n• Regiones de alta/baja actividad",
            "deriv_3d": "🎯 Derivadas 3D\n\nVisualizaci��n tridimensional\nde velocidades de cambio.\n\nInteracciones:\n• Rotación 3D completa\n• Análisis de flujos dinámicos"
        }
        
        info_text = info_texts.get(plot_type, "Selecciona un tipo de visualización")
        
        info_label = ttk.Label(self.viz_info_frame, text=info_text,
                              font=('Arial', 10), justify=tk.CENTER)
        info_label.pack(expand=True)
        
        # Botón para abrir
        if hasattr(self, 'current_figure'):
            open_btn = ttk.Button(self.viz_info_frame, text="🌐 Abrir Visualización",
                                 command=self.open_in_browser)
            open_btn.pack(pady=10)
        
    def open_in_browser(self):
        """Abrir visualización en navegador"""
        
        if not hasattr(self, 'current_figure'):
            messagebox.showwarning("Advertencia", "No hay visualización generada")
            return
            
        try:
            # Crear archivo temporal
            temp_file = os.path.join(tempfile.gettempdir(), "neural_analysis_plot.html")
            self.current_figure.write_html(temp_file)
            
            # Abrir en navegador
            webbrowser.open(f'file://{temp_file}')
            
            self.update_status("Visualización abierta en navegador")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error abriendo en navegador:\n{str(e)}")
            
    def export_plot(self):
        """Exportar gráfico"""
        
        if not hasattr(self, 'current_figure'):
            messagebox.showwarning("Advertencia", "No hay visualización para exportar")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                title="Exportar gráfico",
                defaultextension=".html",
                filetypes=[
                    ("HTML files", "*.html"),
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf")
                ]
            )
            
            if file_path:
                if file_path.endswith('.html'):
                    self.current_figure.write_html(file_path)
                elif file_path.endswith('.png'):
                    self.current_figure.write_image(file_path)
                elif file_path.endswith('.pdf'):
                    self.current_figure.write_image(file_path)
                    
                messagebox.showinfo("Éxito", f"Gráfico exportado a:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exportando:\n{str(e)}")
            
    def reset_controls(self):
        """Resetear controles"""
        
        self.pc_x.set(1)
        self.pc_y.set(2)
        self.pc_z.set(3)
        self.behavior_filter.set("All")
        self.show_derivatives.set(False)
        self.color_scheme.set("viridis")
        self.show_trajectory.set(True)
        self.show_points.set(True)
        self.point_size.set(4)
        
        self.update_status("Controles reseteados")
        
    def run(self):
        """Ejecutar aplicación"""
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Aplicación cerrada por usuario")


def main():
    """Función principal"""
    
    print("🧠 Iniciando C. elegans Neural Analysis Desktop App...")
    
    try:
        app = NeuralAnalysisGUI()
        app.run()
    except Exception as e:
        print(f"Error iniciando aplicación: {str(e)}")
        

if __name__ == "__main__":
    main()