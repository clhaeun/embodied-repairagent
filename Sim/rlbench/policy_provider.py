# RepairAgent policy_provider
# - Provider는 prompt를 만들지 않고 "준비물(inputs)"만 서버로 전송
# - failure: type="repair" (서버가 run_skill 전체를 재작성하도록 유도)
# - not_finished: type="revise" + last_step.need_more_steps=True (서버가 추가 생성/완성하도록 유도)

import os
import json
import threading
from typing import Any, Dict, List, Optional

from ast_policy import sha1
from remote_llm_socket_client import RemoteLLMSocketClient

_DEBUG_RPC = os.environ.get("RPC_DEBUG", "0") == "1"

def _debug_print_req(prefix: str, obj: dict):
    if not _DEBUG_RPC:
        return
    try:
        print(f"[DEBUG_RPC] {prefix}: {json.dumps(obj, ensure_ascii=False)[:2000]}")
    except Exception:
        print(f"[DEBUG_RPC] {prefix}: (failed to dump)")

"""
        if self.mode == "remote_llm":
            # RemoteLLMSocketClient가 timeout kwargs를 지원하는 경우를 대비한 안전 처리
            try:
                self.client = RemoteLLMSocketClient(
                    remote_host,
                    remote_port,
                    connect_timeout=30.0,
                    recv_timeout=600.0
                )
            except TypeError:
                self.client = RemoteLLMSocketClient(remote_host, remote_port)

            print(f"[INFO] connecting to remote LLM server at {remote_host}:{remote_port}")
            self.client.connect()
            print("[INFO] connected.")

        # oracle(reference) 코드 로드 (초기 생성에만 사용/또는 서버가 원하면 사용)
        self.reference_code = self._read(self.source_robot, self.task_name)

    def close(self):
        if self.client:
            self.client.close()
            self.client = None"""


class PolicyProvider:
    """
    - Provider는 prompt를 구성하지 않음
    - 서버가 rtype(repair/revise/generate)에서 get_*_prompt로 최종 prompt 구성 후 LLM 호출
    """

    def __init__(
        self,
        mode: str,                 # "static"|"local_llm"|"remote_llm"
        policy_path: str,
        task_name: str,
        source_robot: str,
        target_robot: str,
        model: str,
        initial_prompt: str,
        remote_host: str = "127.0.0.1",
        remote_port: int = 9000,
        # object_names: Optional[List[str]] = None,
        object_names: list = None,
        descriptions: str = "",
        # target_scene_info: Any = None,
        target_scene_info = None,
        grasp_guidance: str = "",
    ):
        self.mode = mode
        self.policy_path = policy_path
        self.target_policy_path = f"./tasks_{target_robot}/{task_name}.py"
        self.task_name = task_name
        self.source_robot = source_robot
        self.target_robot = target_robot
        self.model = model
        self.initial_prompt = initial_prompt
        self.object_names = object_names or []
        self.descriptions = descriptions
        # self.target_scene_info = target_scene_info or []
        self.target_scene_info = target_scene_info
        self.grasp_guidance = grasp_guidance  # alignment가 필요한 graspable objects 정보
        self.last_response_num_tokens: Optional[int] = None  # 마지막 LLM 응답의 토큰 수
        self.last_prompt: Optional[str] = None  # 마지막 LLM에 전달된 프롬프트
        self.last_hypothesis_prompt: Optional[str] = None  # 마지막 hypothesis 생성용 프롬프트
        self.last_hypothesis_response: Optional[str] = None  # 마지막 hypothesis LLM 응답

        self.client: Optional[RemoteLLMSocketClient] = None
        self.last_response_num_tokens: Optional[int] = None

        self.client: Optional[RemoteLLMSocketClient] = None
        self.bg_client: Optional[RemoteLLMSocketClient] = None  # 백그라운드 스레드용 별도 클라이언트
        self._bg_client_lock = threading.Lock()  # bg_client 접근 보호용 lock

        self._remote_host = remote_host
        self._remote_port = remote_port
        if self.mode == "remote_llm":
            # connect_timeout을 30초로 줄여서 빠르게 실패하도록
            # recv_timeout은 600초(10분)로 LLM 생성 시간 고려
            self.client = RemoteLLMSocketClient(
                remote_host,
                remote_port,
                connect_timeout=30.0,  # 연결 시도는 30초만
                recv_timeout=600.0     # 응답 대기는 10분
            )
            print(f"[INFO] Connecting to remote LLM server at {remote_host}:{remote_port}...")
            self.client.connect()
            print(f"[INFO] Successfully connected to {remote_host}:{remote_port}")
        
        self.reference_code = self._read(self.source_robot, self.task_name)

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
        if self.bg_client:
            self.bg_client.close()
            self.bg_client = None

    # -------------------------
    # Initial code generation
    # -------------------------
    def get_initial_code(self) -> str:
        """
        generate 요청에서:
          - prompt는 initial_prompt 그대로 보내되,
          - reference_code/scene info 등 "재료"를 별도 필드로 함께 보냄
        (서버가 원하면 조립해서 사용)
        """
        if self.mode == "static":
            with open(self.policy_path, "r", encoding="utf-8") as f:
                return f.read()

        if self.mode == "local_llm":

            from generate import generate_code_with_local_llm
            code = generate_code_with_local_llm(self.initial_prompt)
            self._save(code)
            return code

        if self.mode == "remote_llm":
            if self.client is None:
                raise RuntimeError("RemoteLLM client is None. Check server is running and mode=='remote_llm'.")
            assert self.client is not None
            assert self.reference_code is not None

            req = {
                "type": "generate",
                "task_name": self.task_name,
                "robot_type": self.target_robot,
                "prompt": self.initial_prompt,
                "reference_code": self.reference_code,
                "available_objects": self.object_names,
                "descriptions": self.descriptions,
                "source_robot": self.source_robot,
                "target_robot": self.target_robot,
                "skill_name": self.task_name,
                "target_scene_info": self.target_scene_info,
                "grasp_guidance": self.grasp_guidance,
            }
            _debug_print_req("CLIENT->SERVER generate", req)
            resp = self.client.request(req)
            _debug_print_req("CLIENT<-SERVER generate_result", resp)


            if resp.get("type") != "generate_result" or not resp.get("ok", False):
                raise RuntimeError(f"Remote generate failed: {resp}")
            code = resp.get("code", "")
            if not code.strip():
                raise RuntimeError("Remote generate returned empty code.")
            self.last_response_num_tokens = resp.get("num_tokens")
            self._extract_prompt(resp)
            self._save(code)
            return code

        raise ValueError(f"Unknown mode: {self.mode}")

    # -------------------------
    # Step-wise revise (optional)
    # -------------------------
    def revise_on_step(
        self,
        current_code: str,
        executed_stmt_texts: List[str],
        next_index: int,
        scene: Dict[str, Any],
        last_step_payload: Dict[str, Any],
    ) -> Optional[str]:
        """
        step마다 revise를 호출하고 싶을 때 사용하는 인터페이스(필요시 유지).
        여기서 prompt를 만들지 않고 재료만 보냄.
        """
        if self.mode != "remote_llm":
            return None

        assert self.client is not None

        req = {
            "type": "revise",
            "task_name": self.task_name,
            "robot_type": self.target_robot,
            "meta": {"model": self.model},
            "constraints": {
                "must_keep_executed_prefix": True,
                "executed_prefix_count": len(executed_stmt_texts),
                "next_stmt_index": next_index,
            },
            "inputs": {
                "current_code": current_code,
                "executed_stmt_texts": executed_stmt_texts,
                "scene": scene,
                "last_step": last_step_payload,
                "grasp_guidance": self.grasp_guidance,
                "reference_code": self.reference_code,
                # "target_scene_info": self.target_scene_info,
            }
        }
        _debug_print_req("CLIENT->SERVER generate", req)
        resp = self.client.request(req)
        _debug_print_req("CLIENT<-SERVER generate_result", resp)


        if resp.get("type") != "revise_result" or not resp.get("ok", False):
            return None

        new_code = resp.get("code", "")
        if not new_code.strip():
            return None
        if sha1(new_code) == sha1(current_code):
            return None
        self.last_response_num_tokens = resp.get("num_tokens")
        self._extract_prompt(resp)
        self._save(new_code)
        return new_code


    # -------------------------
    # Failure repair (full rewrite expected)
    # -------------------------
    def repair_on_failure(
        self,
        current_code: str,
        executed_stmt_texts: List[str],
        failed_stmt_text: str,
        error_payload: Dict[str, Any],
        scene: Dict[str, Any],
        exploration_obs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        failure 시:
          - Provider는 준비물만 서버로 전달
          - 서버가 get_repair_prompt(...)로 프롬프트를 조립하여
            'run_skill 전체(body 또는 전체 코드)'를 생성하도록 함
        """
        if self.mode == "static":
            raise RuntimeError("static mode cannot repair. Use local_llm or remote_llm.")

        if self.mode == "local_llm":
            from generate import generate_code_with_local_llm
            augmented = (
                self.initial_prompt
                + "\n\n[FAILURE]\n" + json.dumps(error_payload, ensure_ascii=False)
                + "\n\n[SCENE]\n" + json.dumps(scene, ensure_ascii=False)
                + "\n\n[FAILED_STATEMENT]\n" + failed_stmt_text
                + "\n\n[CONSTRAINT]\nDo NOT change already executed prefix statements.\n"
                + "\n[EXECUTED_PREFIX]\n" + "\n---\n".join(executed_stmt_texts)
            )
            new_code = generate_code_with_local_llm(augmented)
            if not new_code.strip():
                raise RuntimeError("local repair returned empty code.")
            self._save(new_code)
            return new_code

        # remote_llm
        assert self.client is not None

        req = {
            "type": "repair",
            "task_name": self.task_name,
            "robot_type": self.target_robot,
            "meta": {"model": self.model},
            "constraints": {
                "must_keep_executed_prefix": True,
                "executed_prefix_count": len(executed_stmt_texts),
            },
            "inputs": {
                "current_code": current_code,
                "executed_stmt_texts": executed_stmt_texts,
                "failed_stmt_text": failed_stmt_text,
                "error": error_payload,
                "scene": scene,
                "grasp_guidance": self.grasp_guidance,
                "reference_code": self.reference_code,
                "exploration_obs": exploration_obs,
            },
        }
        _debug_print_req("CLIENT->SERVER generate", req)
        resp = self.client.request(req)
        _debug_print_req("CLIENT<-SERVER generate_result", resp)

        if resp.get("type") != "repair_result" or not resp.get("ok", False):
            raise RuntimeError(f"Remote repair failed: {resp}")

        new_code = resp.get("code", "")
        if not new_code.strip():
            raise RuntimeError("Remote repair returned empty code.")
        self.last_response_num_tokens = resp.get("num_tokens")
        self._extract_prompt(resp)
        self._save(new_code)
        return new_code

    # -------------------------
    # Not-finished -> revise (server generates additional steps / or returns full code)
    # -------------------------
    def repair_on_not_finished(
        self,
        current_code: str,
        executed_stmt_texts: List[str],
        next_index: int,
        scene: Dict[str, Any],
        last_step_payload: List[Dict[str, Any]],
        exploration_obs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        not_finished(모든 statement 실행했지만 goal 미달) 시:
          - type="revise"로 보내서 서버 revise 경로를 탑니다.
          - last_step.need_more_steps=True를 강제하여 서버가 확실히 "추가 생성" 분기로 동작하도록 합니다.

        서버 구현 권장 계약:
          - revise_result.code는 'continuation(body 일부)'로 반환(서버가 continuation을 current_code에 붙여서 반환)
          - 필요 시 body 전체 반환으로 수정
        """
        if self.mode == "static":
            raise RuntimeError("static mode cannot revise. Use local_llm or remote_llm.")

        if self.mode == "local_llm":
            from generate import generate_code_with_local_llm

            augmented = (
                self.initial_prompt
                + "\n\n[NOT_FINISHED]\nAll statements executed but goal not met.\n"
                + "\n\n[SCENE]\n" + json.dumps(scene, ensure_ascii=False)
                + "\n\n[CONSTRAINT]\nDo NOT change already executed prefix statements.\n"
                + "\n[EXECUTED_PREFIX]\n" + "\n---\n".join(executed_stmt_texts)
                + "\n\n[CURRENT_CODE]\n" + current_code
            )
            new_code = generate_code_with_local_llm(augmented)
            if not new_code.strip():
                raise RuntimeError("local not_finished revise returned empty code.")
            self._save(new_code)
            return new_code

        assert self.client is not None

        last_step_payload = dict(last_step_payload or {})
        last_step_payload["need_more_steps"] = True  # 서버 revise 게이트용

        req = {
            "type": "revise",
            "task_name": self.task_name,
            "robot_type": self.target_robot,
            "meta": {
                "model": self.model,
                "source_robot": self.source_robot,
            },
            "constraints": {
                "must_keep_executed_prefix": True,
                "executed_prefix_count": len(executed_stmt_texts),
                "next_stmt_index": next_index,
            },
            "inputs": {
                "current_code": current_code,
                "executed_stmt_texts": executed_stmt_texts,
                "scene": scene,
                "last_step": last_step_payload,
                "grasp_guidance": self.grasp_guidance,
                "reference_code": self.reference_code,
                "exploration_obs": exploration_obs,
                # "target_scene_info": self.target_scene_info,
            },
            # "gen": {"do_sample": False, "temperature": 0.2, "top_p": 0.9, "min_new_tokens": 200, "max_new_tokens": 900},
        }
        _debug_print_req("CLIENT->SERVER generate", req)
        resp = self.client.request(req)
        _debug_print_req("CLIENT<-SERVER generate_result", resp)

        if resp.get("type") != "revise_result" or not resp.get("ok", False):
            raise RuntimeError(f"Remote revise(not_finished) failed: {resp}")

        new_code = resp.get("code", "")
        if not new_code.strip():
            raise RuntimeError("Remote revise(not_finished) returned empty code.")
        self.last_response_num_tokens = resp.get("num_tokens")
        self._extract_prompt(resp)
        self._save(new_code)
        return new_code

    # -------------------------
    # utils
    # -------------------------
    """
    def _update_token_stats(self, resp: Dict[str, Any]) -> None:
        # SkillCache는 num_tokens를 쓰기도 하고, 다른 서버는 meta.new_tokens를 쓰기도 함
        n = resp.get("num_tokens", None)
        if n is None:
            meta = resp.get("meta") or {}
            n = meta.get("new_tokens", None)
        if isinstance(n, int):
            self.last_response_num_tokens = n
    """

    def _extract_prompt(self, resp: Dict[str, Any]) -> None:
        meta = resp.get("meta") or {}
        prompt = meta.get("prompt")
        if prompt:
            self.last_prompt = prompt
        # hypothesis 정보도 추출
        hyp_prompt = meta.get("hypothesis_prompt")
        hyp_response = meta.get("hypothesis_response")
        self.last_hypothesis_prompt = hyp_prompt if hyp_prompt else None
        self.last_hypothesis_response = hyp_response if hyp_response else None

    def _save(self, code: str) -> None:
        os.makedirs(os.path.dirname(self.target_policy_path) or ".", exist_ok=True)
        with open(self.target_policy_path, "w", encoding="utf-8") as f:
            f.write(code)

    def _read(self, robot_name: str, task_name: str) -> str:
        candidates = [
            f"./common_rlbench/tasks/tasks_{robot_name}/{task_name}.py",  # oracle 폴더(우선)
            f"./tasks_{robot_name}/{task_name}.py",
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
        raise FileNotFoundError(f"Reference policy not found. Tried: {candidates}")
