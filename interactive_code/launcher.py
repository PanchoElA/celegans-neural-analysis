"""
🚀 Launcher para la Aplicación de Análisis Neural
Quick Start - C. elegans Neural Analysis Desktop App
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

def check_dependencies():
    """Verificar dependencias necesarias"""
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('plotly', 'plotly'),
        ('sklearn', 'scikit-learn'),
        ('scipy', 'scipy')
    ]
    
    missing = []
    
    for package, pip_name in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(pip_name)
    
    return missing

def install_packages(packages):
    """Instalar paquetes faltantes"""
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"❌ Error instalando {package}")
            return False
    
    return True

def launch_app():
    """Lanzar la aplicación principal"""
    
    try:
        # Verificar que el archivo existe
        app_file = 'neural_desktop_app_v2.py'
        if not os.path.exists(app_file):
            messagebox.showerror("Error", f"No se encuentra el archivo: {app_file}")
            return
        
        # Lanzar aplicación
        subprocess.Popen([sys.executable, app_file])
        print("🧠 Aplicación lanzada exitosamente!")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error lanzando aplicación: {str(e)}")

def create_launcher_gui():
    """Crear GUI del launcher"""
    
    root = tk.Tk()
    root.title("🧠 Neural Analysis Launcher")
    root.geometry("500x400")
    root.configure(bg='#f0f8ff')
    
    # Título principal
    title_frame = tk.Frame(root, bg='#f0f8ff')
    title_frame.pack(pady=20)
    
    title_label = tk.Label(title_frame, 
                          text="🧠 C. elegans Neural Analysis",
                          font=('Arial', 18, 'bold'),
                          bg='#f0f8ff', fg='#2c3e50')
    title_label.pack()
    
    subtitle_label = tk.Label(title_frame,
                             text="Interactive Desktop Application",
                             font=('Arial', 12),
                             bg='#f0f8ff', fg='#7f8c8d')
    subtitle_label.pack()
    
    # Frame de información
    info_frame = tk.Frame(root, bg='#f0f8ff')
    info_frame.pack(pady=20, padx=40, fill='both', expand=True)
    
    info_text = """
✨ CARACTERÍSTICAS PRINCIPALES:

🔬 Análisis PCA Interactivo
   • PC vs Tiempo, PC vs PC, PCA 3D
   • Visualizaciones completamente interactivas
   
🌊 Análisis de Derivadas
   • Derivadas del firing rate
   • Espacios 2D y 3D de velocidades
   
🎭 Análisis Comportamental
   • Identificación automática de comportamientos
   • Filtrado por Forward/Reverse/Pause/Turn
   
⚙️ Preprocesamiento Avanzado
   • Múltiples métodos de escalado
   • Filtrado temporal configurable
   
🎨 Personalización Completa
   • Esquemas de colores
   • Tamaños y estilos de marcadores
   • Control total de visualización

🌐 Gráficos Interactivos
   • Rotación 3D con mouse
   • Zoom y pan dinámicos
   • Exportación a múltiples formatos
    """
    
    info_label = tk.Label(info_frame, text=info_text,
                         font=('Courier', 9),
                         bg='#f0f8ff', fg='#34495e',
                         justify='left', anchor='nw')
    info_label.pack(fill='both', expand=True)
    
    # Frame de botones
    button_frame = tk.Frame(root, bg='#f0f8ff')
    button_frame.pack(pady=20)
    
    # Botón principal
    launch_btn = tk.Button(button_frame,
                          text="🚀 LANZAR APLICACIÓN",
                          font=('Arial', 14, 'bold'),
                          bg='#3498db', fg='white',
                          padx=30, pady=10,
                          command=launch_app,
                          relief='raised', bd=3)
    launch_btn.pack(pady=10)
    
    # Botón verificar dependencias
    check_btn = tk.Button(button_frame,
                         text="🔍 Verificar Dependencias",
                         font=('Arial', 10),
                         bg='#95a5a6', fg='white',
                         padx=20, pady=5,
                         command=check_and_install_deps)
    check_btn.pack(pady=5)
    
    # Info de archivos
    files_info = tk.Label(root,
                         text="📁 Asegúrate de tener 'neural_data_dataframe.csv' en el directorio",
                         font=('Arial', 9), bg='#f0f8ff', fg='#7f8c8d')
    files_info.pack(pady=10)
    
    return root

def check_and_install_deps():
    """Verificar e instalar dependencias"""
    
    missing = check_dependencies()
    
    if not missing:
        messagebox.showinfo("Dependencias", "✅ Todas las dependencias están instaladas!")
        return
    
    response = messagebox.askyesno(
        "Dependencias Faltantes",
        f"Faltan las siguientes dependencias:\n\n" + 
        "\n".join(f"• {pkg}" for pkg in missing) +
        f"\n\n¿Deseas instalarlas automáticamente?"
    )
    
    if response:
        try:
            success = install_packages(missing)
            if success:
                messagebox.showinfo("Éxito", "✅ Todas las dependencias instaladas correctamente!")
            else:
                messagebox.showerror("Error", "❌ Error instalando algunas dependencias")
        except Exception as e:
            messagebox.showerror("Error", f"Error en instalación: {str(e)}")

def main():
    """Función principal del launcher"""
    
    print("🧠 Neural Analysis Launcher v1.0")
    print("=" * 40)
    
    # Crear GUI
    root = create_launcher_gui()
    
    # Manejar cierre
    def on_closing():
        print("👋 Launcher cerrado")
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Ejecutar
    root.mainloop()

if __name__ == "__main__":
    main()