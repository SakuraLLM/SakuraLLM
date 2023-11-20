from collections import namedtuple
import sys
import os

from PySide6 import QtCore, QtWidgets


def list_modal():
    def is_gpu_modal(f):
        return os.path.isdir(f)

    def is_cpu_modal(f):
        return os.path.isfile(f) and f.endswith(".gguf")

    Modal = namedtuple("Modal", ["name", "device", "button"])

    files = os.listdir()
    gpu_modals = [
        Modal(f, "GPU", QtWidgets.QRadioButton(f"GPU {f}"))
        for f in files
        if is_gpu_modal(f)
    ]
    cpu_modals = [
        Modal(f, "CPU", QtWidgets.QRadioButton(f"CPU {f}"))
        for f in files
        if is_cpu_modal(f)
    ]
    return gpu_modals + cpu_modals


class Example(QtWidgets.QWidget):
    server = None

    def __init__(self):
        super().__init__()

        vbox = QtWidgets.QVBoxLayout()

        self.modals = list_modal()
        for index, modal in enumerate(self.modals):
            modal.button.setChecked(index == 0)
            vbox.addWidget(modal.button)

        self.toggle_button = QtWidgets.QPushButton("启动")
        self.toggle_button.clicked.connect(self.toggleProcess)
        vbox.addStretch(1)
        vbox.addWidget(self.toggle_button)

        self.process = QtCore.QProcess()
        self.process.stateChanged.connect(self.onProcessStateChanged)
        self.process.readyReadStandardError.connect(self.onReadyReadStandardError)
        self.process.readyReadStandardOutput.connect(self.onReadyReadStandardOutput)
        self.terminal = QtWidgets.QTextBrowser()

        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addWidget(self.terminal)

        self.setLayout(hbox)

        self.setGeometry(300, 300, 800, 250)
        self.setWindowTitle("Sakrua Api 启动器")
        self.show()

    @QtCore.Slot()
    def onReadyReadStandardError(self):
        error = self.process.readAllStandardError().data().decode()
        self.terminal.setTextColor(QtCore.Qt.red)
        self.terminal.insertPlainText(error)

    @QtCore.Slot()
    def onReadyReadStandardOutput(self):
        output = self.process.readAllStandardOutput().data().decode()
        self.terminal.setTextColor(self.terminal.palette().text().color())
        self.terminal.insertPlainText(output)

    @QtCore.Slot()
    def onProcessStateChanged(self, state):
        running = state != QtCore.QProcess.NotRunning
        self.toggle_button.setText("关闭" if running else "启动")
        for modal in self.modals:
            modal.button.setDisabled(running)

    @QtCore.Slot()
    def toggleProcess(self):
        state = self.process.state()
        if state == QtCore.QProcess.NotRunning:
            selected_modal = None
            for modal in self.modals:
                if modal.button.isChecked():
                    selected_modal = modal
            if selected_modal:
                self.terminal.clear()
                self.process.startCommand(
                    [
                        "python",
                        "server_fake.py",
                        "--model_name_or_path ./{selected_modal.name}",
                        "--llama_cpp" if selected_modal.device == "CPU" else "",
                        "--model_version 0.8",
                        "--trust_remote_code",
                        "--no-auth",
                        # --use_gptq_model
                        # --log info
                    ].join(" ")
                )
                self.toggle_button.setText("关闭")
            else:
                raise RuntimeError("没有选中模型")
        else:
            self.process.kill()
            self.toggle_button.setText("启动")


def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
