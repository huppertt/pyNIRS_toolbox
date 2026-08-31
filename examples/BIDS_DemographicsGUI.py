import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTabWidget, QPushButton


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("fNIRS BIDS Demographics Editor")
        self.setGeometry(100, 100, 800, 800)
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)

        # Create a QTabWidget in Left panel
        self.tab_widget = QTabWidget()
        left_panel.addWidget(self.tab_widget)
        # Create individual tabs (QWidgets)
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        # Add tabs to the QTabWidget
        self.tab_widget.addTab(self.tab1, "Directory View")
        self.tab_widget.addTab(self.tab2, "Table View")

        # Populate Tab 1
        tab1_layout = QVBoxLayout()
        self.tab1.setLayout(tab1_layout)
        tab1_layout.addWidget(QLabel("Content for Tab 1"))
        tab1_layout.addWidget(QPushButton("Button in Tab 1"))

        # Populate Tab 2
        tab2_layout = QVBoxLayout()
        self.tab2.setLayout(tab2_layout)
        tab2_layout.addWidget(QLabel("This is Tab 2's content."))

    def on_button_click(self):
        self.label.setText("Button Clicked!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
