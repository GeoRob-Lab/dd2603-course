"""gRPC server that makes the simulation look like the real RB-Y1 to rby1-sdk.

It implements the robot's API (rby1-sdk's ``.proto`` files) on top of the simulation, so
``rby1_sdk.create_robot("localhost:50051", "a")`` and everything built on it work
unchanged against ``rby1-sim``, Rainbow Robotics' docker simulator and the real robot.
Services the course does not need (Wi-Fi, LEDs, logs, serial ports, ...) are left out;
calling them raises a gRPC "unimplemented" error in the client.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent import futures

import grpc
import numpy as np

from rbylearn.sim.models import import_rby1_api
from rbylearn.sim.control import Feedback

api = import_rby1_api()
from rb.api import (  # noqa: E402
    control_manager_pb2, control_manager_service_pb2_grpc, parameter_pb2, parameter_service_pb2_grpc,
    ping_pb2, ping_service_pb2_grpc, power_pb2, power_service_pb2_grpc, robot_command_pb2,
    robot_command_service_pb2_grpc, robot_info_pb2, robot_info_service_pb2_grpc, robot_state_pb2,
    robot_state_service_pb2_grpc,
)

PORT = 50051  # the real robot's port
POWER_DEVICES = ("5v", "12v", "24v", "48v")
CMState = control_manager_pb2.ControlManagerState


class SimRobot:
    """Everything the services share: the simulation, its control manager and a lock.

    The physics loop must hold ``lock`` while it steps the simulation.
    """

    def __init__(self, sim, control, robot_model, urdf: str):
        self.sim, self.control, self.robot_model, self.urdf = sim, control, robot_model, urdf
        self.lock = threading.Lock()
        self.parameters: dict[str, str] = {}

    def control_manager_state(self):
        c = self.control
        state = CMState.CONTROL_MANAGER_STATE_ENABLED if c.enabled else CMState.CONTROL_MANAGER_STATE_IDLE
        executing = CMState.CONTROL_STATE_EXECUTING if c.executions else CMState.CONTROL_STATE_IDLE
        return CMState(state=state, time_scale=c.time_scale, control_state=executing,
                       enabled_joint_idx=range(len(self.sim.joint_names)) if c.enabled else [])

    def robot_state(self):
        sim, c = self.sim, self.control
        q, qd, tau, q_target = sim.q(), sim.qd(), sim.joint_torques(), sim.joint_targets()
        n = len(q)
        state = robot_state_pb2.RobotState(
            position=q, velocity=qd, current=np.zeros(n), torque=tau, target_position=q_target,
            target_velocity=np.zeros(n), target_feedback_gain=[0] * n, target_feedforward_torque=np.zeros(n),
            is_ready=[float(c.servo_on)] * n, temperature=[30] * n,
        )
        state.timestamp.FromNanoseconds(time.time_ns())
        on = robot_state_pb2.PowerState.STATE_POWER_ON if c.power_on else robot_state_pb2.PowerState.STATE_POWER_OFF
        for _ in POWER_DEVICES:
            state.power_states.add(state=on, voltage=48.0 if c.power_on else 0.0)
        for k in range(n):
            js = state.joint_states.add(
                is_ready=c.servo_on, power_on=c.power_on, position=q[k], velocity=qd[k], torque=tau[k],
                target_position=q_target[k], temperature=30,
            )
            js.time_since_last_update.FromMilliseconds(2)
        for side in ("right", "left"):
            for message in (getattr(state, f"ft_sensor_{side}"), getattr(state, f"tool_flange_{side}")):
                message.time_since_last_update.FromMilliseconds(2)
        return state


class PingService(ping_service_pb2_grpc.PingServiceServicer):
    def Ping(self, request, context):
        return ping_pb2.PingResponse()


class RobotInfoService(robot_info_service_pb2_grpc.RobotInfoServiceServicer):
    def __init__(self, robot: SimRobot):
        self.robot = robot

    def GetRobotInfo(self, request, context):
        m = self.robot.robot_model
        info = robot_info_pb2.RobotInfo(
            robot_version="rby1-sim", version="rby1-sim", sdk_version="0.10.0", robot_model_name=m.model_name,
            degree_of_freedom=m.robot_dof, mobility_joint_idx=m.mobility_idx, body_joint_idx=m.body_idx,
            head_joint_idx=m.head_idx, torso_joint_idx=m.torso_idx, right_arm_joint_idx=m.right_arm_idx,
            left_arm_joint_idx=m.left_arm_idx,
        )
        for name in POWER_DEVICES:
            info.power_infos.add(name=name)
        for name in m.robot_joint_names:
            info.joint_infos.add(name=name, product_name="rby1-sim")
        return robot_info_pb2.GetRobotInfoResponse(robot_info=info)

    def GetRobotModel(self, request, context):
        return robot_info_pb2.GetRobotModelResponse(model=self.robot.urdf)


class RobotStateService(robot_state_service_pb2_grpc.RobotStateServiceServicer):
    def __init__(self, robot: SimRobot):
        self.robot = robot

    def GetRobotState(self, request, context):
        with self.robot.lock:
            return robot_state_pb2.GetRobotStateResponse(
                robot_state=self.robot.robot_state(), control_manager_state=self.robot.control_manager_state())

    def GetRobotStateStream(self, request, context):
        period = 1.0 / request.update_rate if request.update_rate > 0 else 0.01
        while context.is_active():
            with self.robot.lock:
                response = robot_state_pb2.GetRobotStateStreamResponse(
                    robot_state=self.robot.robot_state(), control_manager_state=self.robot.control_manager_state())
            yield response
            time.sleep(period)

    def GetControlManagerState(self, request, context):
        with self.robot.lock:
            return robot_state_pb2.GetControlManagerStateResponse(control_manager_state=self.robot.control_manager_state())


class PowerService(power_service_pb2_grpc.PowerServiceServicer):
    def __init__(self, robot: SimRobot):
        self.robot = robot

    def PowerCommand(self, request, context):
        with self.robot.lock:
            c = self.robot.control
            c.power_on = request.command == power_pb2.PowerCommandRequest.COMMAND_POWER_ON
            if not c.power_on:
                c.servo_on = c.enabled = False
                c.cancel()
        return power_pb2.PowerCommandResponse(status=power_pb2.PowerCommandResponse.STATUS_SUCCESS)

    def JointCommand(self, request, context):
        with self.robot.lock:
            c = self.robot.control
            ok = c.power_on or request.command != power_pb2.JointCommandRequest.COMMAND_SERVO_ON
            if request.command == power_pb2.JointCommandRequest.COMMAND_SERVO_ON and c.power_on:
                c.servo_on = True
        status = power_pb2.JointCommandResponse.STATUS_SUCCESS if ok else power_pb2.JointCommandResponse.STATUS_INTERNAL_ERROR
        return power_pb2.JointCommandResponse(status=status, message="" if ok else "power is off")


class ControlManagerService(control_manager_service_pb2_grpc.ControlManagerServiceServicer):
    def __init__(self, robot: SimRobot):
        self.robot = robot

    def ControlManagerCommand(self, request, context):
        Command = control_manager_pb2.ControlManagerCommandRequest
        with self.robot.lock:
            c = self.robot.control
            if request.command == Command.COMMAND_ENABLE:
                c.enabled = c.power_on and c.servo_on
                if c.enabled:
                    c.hold_current_pose()
            elif request.command in (Command.COMMAND_DISABLE, Command.COMMAND_RESET_FAULT):
                c.enabled = False
                c.cancel()
            return control_manager_pb2.ControlManagerCommandResponse(control_manager_state=self.robot.control_manager_state())

    def CancelControl(self, request, context):
        with self.robot.lock:
            self.robot.control.cancel()
        return control_manager_pb2.CancelControlResponse()

    def GetTimeScale(self, request, context):
        return control_manager_pb2.GetTimeScaleResponse(time_scale=self.robot.control.time_scale)

    def SetTimeScale(self, request, context):
        with self.robot.lock:
            self.robot.control.time_scale = min(max(request.time_scale, 0.0), 1.0)
            return control_manager_pb2.SetTimeScaleResponse(current_time_scale=self.robot.control.time_scale)

    def WaitForControlReady(self, request, context):
        timeout = request.timeout.seconds + request.timeout.nanos * 1e-9 if request.HasField("timeout") else None
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.robot.control.executions and (deadline is None or time.monotonic() < deadline):
            time.sleep(0.005)
        return control_manager_pb2.WaitForControlReadyResponse(ready=not self.robot.control.executions)


class ParameterService(parameter_service_pb2_grpc.ParameterServiceServicer):
    """Stores parameters so clients can set and read them; the simulation does not use them."""

    def __init__(self, robot: SimRobot):
        self.robot = robot

    def SetParameter(self, request, context):
        try:
            value = json.dumps(json.loads(request.parameter)["value"])
        except (ValueError, KeyError, TypeError):
            value = request.parameter
        self.robot.parameters[request.name] = value
        return parameter_pb2.SetParameterResponse()

    def GetParameter(self, request, context):
        return parameter_pb2.GetParameterResponse(parameter=self.robot.parameters.get(request.name, ""))

    def GetParameterList(self, request, context):
        response = parameter_pb2.GetParameterListResponse()
        for name in self.robot.parameters:
            response.parameters.add(name=name)
        return response


class RobotCommandService(robot_command_service_pb2_grpc.RobotCommandServiceServicer):
    def __init__(self, robot: SimRobot):
        self.robot = robot

    def RobotCommand(self, request, context):
        with self.robot.lock:
            execution = self.robot.control.submit(request.robot_command, request.priority)
        while not execution.done.wait(0.01):
            if not context.is_active():  # the client cancelled the command
                with self.robot.lock:
                    execution.finish(Feedback.FINISH_CODE_CANCELED)
                    if execution in self.robot.control.executions:
                        self.robot.control.executions.remove(execution)
        return robot_command_pb2.RobotCommandResponse(feedback=execution.feedback())

    def RobotCommandStream(self, request_iterator, context):
        execution = None
        try:
            for request in request_iterator:
                with self.robot.lock:
                    if request.robot_command.WhichOneof("command"):  # a new command; an empty one asks for feedback
                        execution = self.robot.control.submit(request.robot_command, request.priority)
                    feedback = execution.feedback() if execution else robot_command_pb2.RobotCommand.Feedback()
                yield robot_command_pb2.RobotCommandResponse(feedback=feedback)
        finally:  # the client closed or cancelled the stream
            if execution is not None:
                with self.robot.lock:
                    execution.finish(Feedback.FINISH_CODE_CANCELED)
                    if execution in self.robot.control.executions:
                        self.robot.control.executions.remove(execution)


def start_server(robot: SimRobot, port: int = PORT) -> grpc.Server:
    """Serve the robot API on ``0.0.0.0:port`` in background threads."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32), options=[("grpc.so_reuseport", 0)])
    for add, servicer in [
        (ping_service_pb2_grpc.add_PingServiceServicer_to_server, PingService()),
        (robot_info_service_pb2_grpc.add_RobotInfoServiceServicer_to_server, RobotInfoService(robot)),
        (robot_state_service_pb2_grpc.add_RobotStateServiceServicer_to_server, RobotStateService(robot)),
        (power_service_pb2_grpc.add_PowerServiceServicer_to_server, PowerService(robot)),
        (control_manager_service_pb2_grpc.add_ControlManagerServiceServicer_to_server, ControlManagerService(robot)),
        (parameter_service_pb2_grpc.add_ParameterServiceServicer_to_server, ParameterService(robot)),
        (robot_command_service_pb2_grpc.add_RobotCommandServiceServicer_to_server, RobotCommandService(robot)),
    ]:
        add(servicer, server)
    try:
        bound = server.add_insecure_port(f"0.0.0.0:{port}")
    except RuntimeError:  # recent grpcio raises instead of returning 0
        bound = 0
    if bound == 0:
        raise SystemExit(f"Port {port} is already in use by another program. Start the simulator on another "
                         f"port, e.g. `rby1-sim --port {port + 1}`, and connect to localhost:{port + 1}.")
    server.start()
    return server
