from __future__ import annotations

import io
import time

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout


class TerminalPanel(QGroupBox):
    def __init__(self, title: str = "Terminal") -> None:
        super().__init__(title)

        root = QVBoxLayout()

        controls = QHBoxLayout()
        controls.addStretch(1)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear)
        controls.addWidget(clear_button)
        root.addLayout(controls)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.setPlaceholderText("Command output will appear here.")
        root.addWidget(self.output)

        self.setLayout(root)

    def append_text(self, text: str) -> None:
        if not text:
            return
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def clear(self) -> None:
        self.output.clear()


class TerminalStream(io.TextIOBase):
    def __init__(self, panel: TerminalPanel, process_events_interval_s: float = 0.05) -> None:
        super().__init__()
        self._panel = panel
        self._interval = process_events_interval_s
        self._last_process = 0.0

    def write(self, text: str) -> int:
        self._panel.append_text(text)

        now = time.monotonic()
        if now - self._last_process >= self._interval:
            QApplication.processEvents()
            self._last_process = now

        return len(text)

    def flush(self) -> None:
        QApplication.processEvents()
