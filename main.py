from src.ui import create_ui

if __name__ == "__main__":
    demo = create_ui()
    # Chạy cục bộ. Nếu muốn share link public thì thêm share=True
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)