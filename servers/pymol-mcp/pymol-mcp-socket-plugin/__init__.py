'''
PyMOL MCP Plugin

A plugin that listens for socket connections and executes PyMOL commands received via socket.
The plugin also provides a basic UI for interaction.

Based on the concept of the "Rendering Plugin" from Michael Lerner.
'''

from __future__ import absolute_import
from __future__ import print_function

import os
import socket
import json
import threading
import time
import traceback

# Global variables
dialog = None
socket_server = None
received_commands = []
listening = False
current_port = 9876  # Default port

def __init_plugin__(app=None):
    '''
    Add an entry to the PyMOL "Plugin" menu
    '''
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('PyMol MCP Socket Plugin', run_plugin_gui)

class SocketServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.socket = None
        self.client = None
        self.running = False
        self.thread = None
        self.command_callback = None
        
    def start(self, command_callback=None):
        """Start the socket server on a separate thread"""
        if self.running:
            return False
            
        self.command_callback = command_callback
        self.running = True
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True  # Daemon thread will exit when main thread exits
        self.thread.start()
        return True
        
    def _run_server(self):
        """Run the socket server in a separate thread"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.settimeout(1.0)  # 1 second timeout for accepting connections
            
            print(f"PyMOL MCP Socket server listening on {self.host}:{self.port}")
            
            while self.running:
                try:
                    # Accept connection (with timeout)
                    self.client, address = self.socket.accept()
                    print(f"Connected to client: {address}")
                    self.client.settimeout(1.0)  # Set timeout for receiving data
                    
                    # Handle client connection
                    buffer = b''
                    while self.running:
                        try:
                            data = self.client.recv(4096)
                            if not data:
                                break  # Connection closed
                                
                            buffer += data
                            
                            # Try to parse JSON
                            try:
                                command = json.loads(buffer.decode('utf-8'))
                                buffer = b''  # Reset buffer
                                
                                # Process the command and get result
                                result = self._handle_command(command)
                                
                                # Send response with result
                                response = json.dumps({
                                    "status": "success", 
                                    "result": result if result else "Command executed"
                                })
                                self.client.sendall(response.encode('utf-8'))
                            except json.JSONDecodeError:
                                # Incomplete JSON, continue receiving
                                continue
                                
                        except socket.timeout:
                            # Socket timeout, just continue the loop
                            continue
                        except Exception as e:
                            print(f"Error receiving data: {str(e)}")
                            break
                    
                    # Close client connection
                    if self.client:
                        self.client.close()
                        self.client = None
                except socket.timeout:
                    # No connection waiting, continue
                    continue
                except Exception as e:
                    print(f"Error accepting connection: {str(e)}")
                    
        except Exception as e:
            print(f"Socket server error: {str(e)}")
            traceback.print_exc()
        finally:
            if self.socket:
                self.socket.close()
            self.running = False
            print("Socket server stopped")
    
    def _handle_command(self, command):
        """Handle received command"""
        if not command:
            return
            
        cmd_type = command.get("type")
        cmd_code = command.get("code", "")
        
        # Save the received command
        global received_commands
        received_commands.append(cmd_code)
        
        # Execute the PyMOL command if a callback is registered
        if self.command_callback and cmd_code:
            result = self.command_callback(cmd_code)
            return result
    
    def stop(self):
        """Stop the socket server"""
        self.running = False
        if self.thread:
            self.thread.join(2.0)  # Wait up to 2 seconds for thread to exit
        if self.client:
            self.client.close()
        if self.socket:
            self.socket.close()
        self.socket = None
        self.client = None
        self.thread = None
        
def run_plugin_gui():
    '''
    Open our custom dialog
    '''
    global dialog
    
    if dialog is None:
        dialog = make_dialog()
    
    dialog.show()

def make_dialog():
    # entry point to PyMOL's API
    from pymol import cmd
    
    # pymol.Qt provides the PyQt5 interface, but may support PyQt4
    # and/or PySide as well
    from pymol.Qt import QtWidgets
    from pymol.Qt.utils import loadUi
    from pymol.Qt.utils import getSaveFileNameWithExt
    
    # create a new Window
    dialog = QtWidgets.QDialog()
    
    # populate the Window from our *.ui file which was created with the Qt Designer
    uifile = os.path.join(os.path.dirname(__file__), 'pymol_mcp_plugin.ui')
    form = loadUi(uifile, dialog)
    
    # Set up socket controls
    form.input_port.setValue(current_port)
    update_status_label(form, "Not listening")
    
    # Define a function to execute PyMOL commands and capture output
    def execute_pymol_command(code):
        try:
            # Execute in PyMOL with output capture
            print(f"Executing PyMOL command from MCP:\n{code}")
            
            # Add code to capture output
            import io
            from contextlib import redirect_stdout
            
            # Prepare the execution environment
            exec_globals = {"cmd": cmd, "__builtins__": __builtins__}
            
            # Capture standard output
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                # Use exec for more complex code that might have functions, etc.
                exec(code, exec_globals)
            
            # Get the captured output
            output = output_buffer.getvalue()
            
            # If we have captured output, return it
            if output:
                print(f"Command output: {output}")
                return {"executed": True, "output": output}
            else:
                # If there's no stdout, check if a result was assigned to a special variable
                if '_result' in exec_globals:
                    result = str(exec_globals['_result'])
                    print(f"Command result: {result}")
                    return {"executed": True, "output": result}
                
                return {"executed": True, "output": "Command executed successfully (no output)"}
        except Exception as e:
            error_msg = f"Error executing PyMOL command: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {"executed": False, "error": error_msg}
    
    # Callback for the "Start Listening" button
    def toggle_listening():
        global socket_server, listening, current_port
        
        if not listening:
            # Start the socket server
            port = form.input_port.value()
            current_port = port
            
            socket_server = SocketServer(port=port)
            if socket_server.start(execute_pymol_command):
                listening = True
                form.button_toggle_listening.setText("Stop Listening")
                update_status_label(form, f"Listening on port {port}")
        else:
            # Stop the socket server
            if socket_server:
                socket_server.stop()
            listening = False
            form.button_toggle_listening.setText("Start Listening")
            update_status_label(form, "Not listening")
    
    # Callback for the "Show Commands" button
    def show_commands():
        global received_commands
        
        if not received_commands:
            cmd.feedback("No commands received yet", "output")
            return
            
        # Print all received commands to the PyMOL console
        cmd.feedback("=== Received Commands ===", "output")
        for i, command in enumerate(received_commands):
            cmd.feedback(f"--- Command {i+1} ---", "output")
            cmd.feedback(command, "output")
        cmd.feedback("=======================", "output")
    
    # Callback for the "Clear Commands" button
    def clear_commands():
        global received_commands
        received_commands = []
        cmd.feedback("Command history cleared", "output")
    
    # Callback for the "Close" button
    def close_dialog():
        global socket_server, listening
        
        # Stop the socket server if it's running
        if socket_server and listening:
            socket_server.stop()
            listening = False
        
        dialog.close()
    
    # Hook up button callbacks
    form.button_toggle_listening.clicked.connect(toggle_listening)
    # form.button_show_commands.clicked.connect(show_commands)
    # form.button_clear_commands.clicked.connect(clear_commands)
    form.button_close.clicked.connect(close_dialog)
    
    return dialog

def update_status_label(form, text):
    """Update the status label with the given text"""
    form.label_status.setText(text)
    
    # Also set a color based on the status
    if "Not listening" in text:
        form.label_status.setStyleSheet("color: red;")
    elif "Listening" in text:
        form.label_status.setStyleSheet("color: green;")
    else:
        form.label_status.setStyleSheet("")