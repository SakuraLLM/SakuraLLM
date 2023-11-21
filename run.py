from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import json
import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets
from ansi2html import Ansi2HTMLConverter


class ModelType(Enum):
    GPTQ = 1
    LLAMA = 2


@dataclass
class Model:
    model_path: str
    model_type: ModelType
    model_name: str
    model_version: str
    model_quant: str


def list_model():
    def parse_model_dir(model_dir):
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
            model_name = config["sakura_name"]
            model_version = config["sakura_version"]
            model_quant = config["sakura_quant"]
            assert isinstance(model_name, str)
            assert isinstance(model_version, str)
            assert isinstance(model_quant, str)
            return Model(
                model_path=model_dir,
                model_type=ModelType.GPTQ,
                model_name=model_name,
                model_version=model_version,
                model_quant=model_quant,
            )

    def parse_model_file(model_file):
        assert model_file.endswith(".gguf")
        return Model(
            model_path=model_file,
            model_type=ModelType.LLAMA,
            model_name="llama_cpp",
            model_version="Unknown",
            model_quant="unknown",
        )

    models: List[Model] = []
    for model_path in os.listdir():
        try:
            if os.path.isdir(model_path):
                model = parse_model_dir(model_path)
                models.append(model)
            elif os.path.isfile(model_path):
                model = parse_model_file(model_path)
                models.append(model)
        except:
            pass

    return models


class NoModelDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(NoModelDialog, self).__init__(parent)
        self.setWindowTitle("My Form")


class ApiServerConsole(QtWidgets.QWidget):
    models: List[Tuple[Model, QtWidgets.QRadioButton]] = []

    def __init__(self):
        super().__init__()

        self.scan_button = QtWidgets.QPushButton("重新扫描模型")
        self.scan_button.clicked.connect(self.scanModels)
        self.toggle_button = QtWidgets.QPushButton("启动")
        self.toggle_button.clicked.connect(self.toggleProcess)

        self.vbox = QtWidgets.QVBoxLayout()
        self.scanModels()
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.scan_button)
        self.vbox.addWidget(self.toggle_button)

        self.process = QtCore.QProcess()
        self.process.stateChanged.connect(self.onProcessStateChanged)
        self.process.readyReadStandardOutput.connect(self.onReadyReadStandardOutput)
        self.process.readyReadStandardError.connect(self.onReadyReadStandardOutput)
        self.terminal = QtWidgets.QTextBrowser()
        # 使用等宽字体
        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.terminal.setFont(fixed_font)
        self.terminal_ansi_converter = Ansi2HTMLConverter()

        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(self.vbox)
        hbox.addWidget(self.terminal)

        self.setLayout(hbox)

        self.setGeometry(300, 300, 800, 250)
        self.setWindowTitle("Sakrua Api 启动器")
        self.show()

    def closeEvent(self, event):
        self.process.kill()
        self.process.waitForFinished()
        event.accept()

    @QtCore.Slot()
    def onReadyReadStandardOutput(self):
        output = self.process.readAllStandardOutput().data().decode()
        html = self.terminal_ansi_converter.convert(
            output,
            full=False,
        )
        print()
        # self.terminal_ansi_converter
        # self.terminal.setStyle(QtWidgets.QStyle.fr)
        self.terminal.insertHtml(html)

    @QtCore.Slot()
    def onProcessStateChanged(self, state):
        running = state != QtCore.QProcess.NotRunning
        self.toggle_button.setText("关闭" if running else "启动")
        for _, button in self.models:
            button.setDisabled(running)
        self.scan_button.setDisabled(running)

    @QtCore.Slot()
    def scanModels(self):
        for _, button in self.models:
            button.deleteLater()

        self.models = []
        for index, model in enumerate(list_model()):
            button = QtWidgets.QRadioButton(model.model_path)
            button.setChecked(index == 0)
            self.vbox.insertWidget(index, button)
            self.models.append((model, button))

    @QtCore.Slot()
    def toggleProcess(self):
        state = self.process.state()
        if state == QtCore.QProcess.NotRunning:
            selected_model = None
            for model, button in self.models:
                if button.isChecked():
                    selected_model = model
            if selected_model is None:
                msgBox = QtWidgets.QMessageBox()
                msgBox.setWindowTitle(" ")
                msgBox.setText("启动失败:没有选中模型")
                msgBox.exec()
                return

            self.terminal.clear()
            self.startApiServer(selected_model)
        else:
            self.closeApiServer()

    def startApiServer(self, model: Model):
        command_parts = [
            "python",
            "server_fake.py",
            f"--model_name_or_path ./{model.model_path}",
            "--trust_remote_code",
            "--no-auth",
            # --log info
        ]
        if model.model_type == ModelType.LLAMA:
            command_parts.append("--model_version 0.8")
            command_parts.append("--llama_cpp")
        elif model.model_type == ModelType.GPTQ:
            command_parts.append("--model_version " + model.model_version)
            if model.model_quant != "":
                command_parts.append("--use_gptq_model")

        self.process.startCommand(" ".join(command_parts))

    def closeApiServer(self):
        self.process.kill()


def main():
    app = QtWidgets.QApplication(sys.argv)
    console = ApiServerConsole()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
