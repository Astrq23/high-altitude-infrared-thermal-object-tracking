from src.ui import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    print("UI is running at http://localhost:7860")
    
#