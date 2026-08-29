Sekarang aku punya semua yang dibutuhkan. Mari mulai.

FILE: factors/proposal.py
Ini file yang berisi implementasi konkret dari "propose hypothesis" dan "convert hypothesis → faktor". Di loop.py, class-class ini di-load secara dinamis via import_class() dari string path di settings.py.

BAGIAN 1: Import (baris 1–18)

import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined
# ^ Jinja2: template engine
#   dipakai untuk render prompt template dengan variabel
#   StrictUndefined: kalau ada variabel yang tidak di-pass → error (bukan silent empty)

from factors.coder.factor import FactorExperiment, FactorTask
# ^ FactorTask: dataclass satu faktor (name, description, formulation, expression, variables)
#   FactorExperiment: alias untuk Experiment (FactorExperiment = Experiment)

from components.proposal import FactorHypothesisGen, FactorHypothesis2Experiment
# ^ parent class — kita telusuri sebentar lagi

from core.prompts import Prompts
# ^ class yang load file YAML → dict

from core.proposal import Hypothesis, Scenario, Trace
# ^ sudah kita bahas sebelumnya

from core.experiment import Experiment
from factors.experiment import QlibFactorExperiment
# ^ subclass Experiment khusus Qlib

from llm.client import LocalLLMBackend, robust_json_parse
# ^ LocalLLMBackend: kirim prompt ke LLM
#   robust_json_parse: parse JSON response LLM (lebih toleran dari json.loads)

import os
import pandas as pd
from log import logger
from factors.regulator.factor_regulator import FactorRegulator
# ^ regulator: cek kualitas ekspresi faktor (duplikasi, kompleksitas, dll)
MASUK KE core/prompts.py — Prompts

class Prompts(SingletonBaseClass, dict[str, str]):
    # ^ inherit dari SingletonBaseClass DAN dict
    #   SingletonBaseClass: hanya satu instance per set argumen
    #   dict: Prompts IS-A dictionary → bisa langsung prompts["key"]

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        with file_path.open(encoding="utf8") as file:
            prompt_yaml_dict = yaml.safe_load(file)
            # ^ baca YAML file → Python dict

        if prompt_yaml_dict is None:
            raise ValueError(f"Failed to load prompts from {file_path}")

        for key, value in prompt_yaml_dict.items():
            self[key] = value
            # ^ copy semua key-value dari YAML ke dict ini
            #   setelah ini, prompts["hypothesis_gen"] → isi template prompt
Intinya: Prompts("path/to/prompts.yaml") → dict yang isinya semua template prompt dari YAML file. Karena SingletonBaseClass, file yang sama hanya dibaca sekali.

MASUK KE components/proposal/init.py — Parent class hierarchy
File ini mendefinisikan rantai class abstrak yang menjembatani core.proposal (sangat abstrak) dengan factors.proposal (implementasi konkret).


prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")
# ^ load template prompt dari components/proposal/prompts.yaml
#   ini prompt GENERIK yang bisa dipakai untuk faktor, model, dll
class LLMHypothesisGen(HypothesisGen) (baris 21–58)

class LLMHypothesisGen(HypothesisGen):
    # ^ inherit dari core.proposal.HypothesisGen (ABC)
    #   menambahkan: logic untuk memanggil LLM

    def __init__(self, scen: Scenario):
        super().__init__(scen)    # simpan self.scen

    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...
    # ^ abstract: subclass harus siapkan dict context + bool json_mode

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...
    # ^ abstract: subclass harus parse response LLM → Hypothesis object
Method gen() — implementasi default:


    def gen(self, trace: Trace) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)
        # ^ subclass siapkan konteks

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                targets=self.targets,          # "factors" atau "model tuning"
                scenario=self.scen.get_scenario_all_desc(
                    filtered_tag="hypothesis_and_experiment"
                ),
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        # ^ render system prompt template dengan variabel:
        #   - targets: apa yang sedang di-generate ("factors")
        #   - scenario: deskripsi lengkap skenario (market, data, interface)
        #   - output_format: format JSON yang diharapkan
        #   - specification: spesifikasi detail hipotesis

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=self.targets,
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                RAG=context_dict["RAG"],       # Retrieval Augmented Generation context (None untuk sekarang)
            )
        )
        # ^ render user prompt: riwayat + feedback dari percobaan sebelumnya

        resp = LocalLLMBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag
        )
        # ^ KIRIM KE LLM!
        #   system_prompt: instruksi tetap (siapa kamu, skenario, format output)
        #   user_prompt: konteks dinamis (riwayat, feedback, direction)
        #   json_mode: paksa LLM output JSON valid

        hypothesis = self.convert_response(resp)
        # ^ parse JSON response → Hypothesis object

        return hypothesis
class FactorHypothesisGen(LLMHypothesisGen) (baris 61–64)

class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"
        # ^ hanya set targets = "factors"
        #   ini dipakai di template: "You are tasked with proposing {targets}..."
class LLMHypothesis2Experiment(Hypothesis2Experiment) (baris 79–110)

class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    # ^ inherit dari core.proposal.Hypothesis2Experiment (ABC)

    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, trace: Trace) -> Experiment: ...

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.get_scenario_all_desc(
                    filtered_tag="hypothesis_and_experiment"
                ),
                experiment_output_format=context["experiment_output_format"],
            )
        )
        # ^ system prompt: skenario + format output faktor

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                target_list=context["target_list"],      # faktor-faktor sebelumnya
                RAG=context["RAG"],
            )
        )
        # ^ user prompt: hipotesis target + riwayat + faktor sebelumnya

        resp = LocalLLMBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag
        )
        return self.convert_response(resp, trace)
class FactorHypothesis2Experiment(LLMHypothesis2Experiment) (baris 113–116)

class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"
Ringkasan hierarki class:


core.proposal.HypothesisGen (ABC)
  └─ components.proposal.LLMHypothesisGen (+ gen() dengan LLM call)
       └─ components.proposal.FactorHypothesisGen (targets="factors")
            └─ factors.proposal.QlibFactorHypothesisGen (implementasi Qlib)
            └─ factors.proposal.AlphaAgentHypothesisGen (implementasi AlphaAgent) ← YANG DIPAKAI

core.proposal.Hypothesis2Experiment (ABC)
  └─ components.proposal.LLMHypothesis2Experiment (+ convert() dengan LLM call)
       └─ components.proposal.FactorHypothesis2Experiment (targets="factors")
            └─ factors.proposal.QlibFactorHypothesis2Experiment (implementasi Qlib)
            └─ factors.proposal.AlphaAgentHypothesis2FactorExpression (implementasi AlphaAgent) ← YANG DIPAKAI
MASUK KE factors/coder/factor.py — FactorTask

class FactorTask(CoSTEERTask):
    # ^ inherit dari CoSTEERTask (task generik untuk coding+evaluation)
    #   menambahkan atribut spesifik faktor

    def __init__(
        self,
        factor_name,            # "vol_mom_ratio"
        factor_description,     # "Ratio of volatility to momentum..."
        factor_formulation,     # "σ(Close,20) / μ(Volume,10)"
        factor_expression=None, # "Std(Close,20)/Mean(Volume,10)"
        *args,
        variables: dict = {},   # {"Close": "Closing price", "Volume": "Trading volume"}
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        self.factor_name = factor_name
        self.factor_description = factor_description
        self.factor_formulation = factor_formulation
        self.factor_expression = factor_expression
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, *args, **kwargs)
        # ^ CoSTEERTask.__init__ mendapat name=factor_name

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""
        # ^ string ini disisipkan ke prompt saat coder perlu generate kode
Dan di bawah file:


FactorExperiment = Experiment
# ^ alias: FactorExperiment hanyalah Experiment
#   ini supaya kode bisa `from factors.coder.factor import FactorExperiment`
#   tanpa harus import core.experiment langsung
MASUK KE factors/experiment.py — QlibFactorExperiment dan QlibAlphaAgentScenario

from rdagent.scenarios.qlib.experiment.factor_experiment import (
    QlibFactorScenario,
    FactorExperiment,
    FactorTask,
    FactorFBWorkspace,
)
# ^ import dari rdagent (framework dasar) — class-class bawaan Qlib experiment

from factors.workspace import QlibFBWorkspace
# ^ workspace custom project ini (bukan rdagent default)
class QlibFactorExperiment (baris 23–33)

class QlibFactorExperiment(_OrigQlibFactorExperiment):
    # ^ override rdagent QlibFactorExperiment

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import rdagent.scenarios.qlib.experiment.factor_experiment as _fe_mod

        rdagent_template_path = Path(_fe_mod.__file__).parent / "factor_template"
        # ^ path ke template folder: file-file template untuk generate kode faktor

        self.experiment_workspace = QlibFBWorkspace(
            template_folder_path=rdagent_template_path
        )
        # ^ GANTI workspace default rdagent dengan QlibFBWorkspace custom
        #   ini memastikan konfigurasi template yang benar dipakai
class QlibAlphaAgentScenario (baris 36–57) — Scenario yang dipakai AlphaAgentLoop
Ingat di settings.py: scen: str = "factors.experiment.QlibAlphaAgentScenario". Ini class-nya.


class QlibAlphaAgentScenario(QlibFactorScenario):
    # ^ inherit dari rdagent QlibFactorScenario

    def __init__(self, use_local: bool = True, *args, **kwargs):
        from rdagent.core.scenario import Scenario
        from factors.qlib_utils import get_data_folder_intro as local_get_data_folder_intro

        Scenario.__init__(self)
        # ^ langsung panggil Scenario.__init__, BUKAN super().__init__()
        #   ini bypass QlibFactorScenario.__init__ yang mungkin setup Docker

        tpl_prefix = "scenarios.qlib.experiment.prompts"
        # ^ prefix template dari rdagent

        self._background = deepcopy(
            T(f"{tpl_prefix}:qlib_factor_background").r(
                runtime_environment=self.get_runtime_environment(),
            )
        )
        # ^ T().r() = load template → render
        #   background: "You are working with Qlib, a quantitative finance library..."

        self._source_data = deepcopy(local_get_data_folder_intro(use_local=use_local))
        # ^ deskripsi data sumber — LOCAL version (bukan Docker)
        #   ini yang membedakan: use_local=True → pakai data lokal

        self._output_format = deepcopy(T(f"{tpl_prefix}:qlib_factor_output_format").r())
        # ^ "Output should be a pickle file containing a DataFrame..."

        self._interface = deepcopy(T(f"{tpl_prefix}:qlib_factor_interface").r())
        # ^ cara menjalankan kode faktor

        self._strategy = deepcopy(T(f"{tpl_prefix}:qlib_factor_strategy").r())
        self._simulator = deepcopy(T(f"{tpl_prefix}:qlib_factor_simulator").r())
        self._rich_style_description = deepcopy(T(f"{tpl_prefix}:qlib_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(f"{tpl_prefix}:qlib_factor_experiment_setting").r())
        # ^ semua deskripsi skenario — ini yang nanti disisipkan ke prompt LLM
        #   saat get_scenario_all_desc() dipanggil
Kembali ke factors/proposal.py

BAGIAN 2: Utility functions (baris 19–54)

DEFAULT_HISTORY_LIMIT = 6
# ^ default: hanya kirim 6 riwayat terakhir ke prompt
#   supaya tidak melampaui batas token LLM

MIN_HISTORY_LIMIT = 1
# ^ minimum: setidaknya 1 riwayat
render_hypothesis_and_feedback() (baris 23–34)

def render_hypothesis_and_feedback(prompt_dict, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> str:
    if len(trace.hist) > 0:
        limited_trace = Trace(scen=trace.scen)
        # ^ buat Trace baru (kosong)

        limited_trace.hist = trace.hist[-history_limit:] if history_limit > 0 else trace.hist
        # ^ copy hanya N riwayat terakhir
        #   misal history_limit=6 dan ada 20 riwayat → ambil 14–19 (6 terakhir)

        return (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=limited_trace)
        )
        # ^ render Jinja2 template "hypothesis_and_feedback" dengan trace terbatas
        #   template ini mengiterasi trace.hist dan format setiap
        #   (hypothesis, experiment, feedback) menjadi teks untuk prompt

    else:
        return "No previous hypothesis and feedback available since it's the first round."
        # ^ round pertama: tidak ada riwayat
is_input_length_error() (baris 37–50)

def is_input_length_error(error_msg: str) -> bool:
    error_indicators = [
        "input length", "context length", "maximum context",
        "token limit", "InvalidParameter", "Range of input length",
        "max_tokens", "too long"
    ]
    error_str = str(error_msg).lower()
    return any(indicator.lower() in error_str for indicator in error_indicators)
    # ^ cek apakah error ini tentang token limit
    #   dipakai untuk retry dengan history_limit yang lebih kecil
    #   misal: prompt terlalu panjang → kurangi riwayat → coba lagi
Alias dan global prompts (baris 53–54, 88, 199)

QlibFactorHypothesis = Hypothesis
# ^ alias: QlibFactorHypothesis = Hypothesis biasa

qa_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "proposal.yaml")
# ^ load prompt template dari factors/prompts/proposal.yaml
#   berisi template untuk hypothesis_gen, hypothesis2experiment, dll

base_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")
# ^ load prompt template lain dari factors/prompts/prompts.yaml
#   dipakai oleh QlibFactorHypothesisGen (bukan AlphaAgent)
BAGIAN 3: class AlphaAgentHypothesis(Hypothesis) (baris 56–86)

class AlphaAgentHypothesis(Hypothesis):
    # ^ extend Hypothesis dengan concise_specification

    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str       # ← BARU: spesifikasi ringkas faktor
    ) -> None:
        super().__init__(
            hypothesis,
            "",                           # reason: kosong (tidak dipakai di AlphaAgent)
            "",                           # concise_reason: kosong
            concise_observation,
            concise_justification,
            concise_knowledge,
        )
        self.concise_specification = concise_specification
        # ^ spesifikasi teknis ringkas tentang faktor yang diusulkan
        #   misal: "Use 20-day rolling window, apply cross-sectional rank"

    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                concise Specification: {self.concise_specification}
                """
BAGIAN 4: class AlphaAgentHypothesisGen (baris 202–321) — YANG DIPAKAI DI PIPELINE
Ini class yang di-load via import_class("factors.proposal.AlphaAgentHypothesisGen") di AlphaAgentLoop.__init__.


class AlphaAgentHypothesisGen(FactorHypothesisGen):
    # ^ inherit: HypothesisGen → LLMHypothesisGen → FactorHypothesisGen → ini
    #   self.targets sudah "factors" dari FactorHypothesisGen

    def __init__(self, scen: Scenario, potential_direction: str=None):
        super().__init__(scen)                   # simpan scen, set targets="factors"
        self.potential_direction = potential_direction
        # ^ direction dari user/planning, misal "momentum cross-sectional"
        #   bisa None kalau user tidak kasih direction
Method prepare_context() (baris 207–229)

    def prepare_context(self, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict, bool]:

        if len(trace.hist) > 0:
            hypothesis_and_feedback = render_hypothesis_and_feedback(
                qa_prompt_dict, trace, history_limit
            )
            # ^ ada riwayat: render riwayat terakhir (max history_limit)

        elif self.potential_direction is not None:
            hypothesis_and_feedback = (
                Environment(undefined=StrictUndefined)
                .from_string(qa_prompt_dict["potential_direction_transformation"])
                .render(potential_direction=self.potential_direction)
            )
            # ^ round pertama + ada direction: render template khusus
            #   template ini mengubah direction jadi "seeding context"
            #   misal: "Your initial research direction is: momentum cross-sectional.
            #   Consider factors related to this area..."

        else:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round. You are encouraged to propose an innovative hypothesis that diverges significantly from existing perspectives."
            # ^ round pertama + tidak ada direction: encourage LLM untuk explore bebas

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,                    # belum ada RAG
            "hypothesis_output_format": qa_prompt_dict["hypothesis_output_format"],
            # ^ format JSON yang diharapkan: {"hypothesis": "...", "concise_observation": "...", ...}
            "hypothesis_specification": qa_prompt_dict["factor_hypothesis_specification"],
            # ^ spesifikasi apa yang harus ada di hipotesis
        }
        return context_dict, True
        # ^ True = json_mode (paksa LLM output JSON)
Method convert_response() (baris 231–244)

    def convert_response(self, response: str) -> AlphaAgentHypothesis:
        response_dict = robust_json_parse(response)
        # ^ parse JSON response LLM
        #   robust_json_parse lebih toleran: handle trailing comma, missing quotes, dll

        hypothesis = AlphaAgentHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_specification=response_dict.get("concise_specification", ""),
        )
        # ^ buat AlphaAgentHypothesis dari JSON
        #   .get("key", "") → aman kalau field tidak ada

        return hypothesis
Method _call_llm() (baris 246–256)

    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        return self.llm_backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt,
            json_mode=json_mode,
        )
        # ^ panggil LLM via self.llm_backend
        #   self.llm_backend di-set oleh AlphaAgentLoop.__init__:
        #     self.hypothesis_generator.llm_backend = llm_backend
        #   bisa LocalLLMBackend (local model) atau API backend
Method gen() (baris 258–321) — OVERRIDE dari LLMHypothesisGen.gen()
Ini yang sebenarnya dipanggil di loop.py step factor_propose().


    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        history_limit = DEFAULT_HISTORY_LIMIT    # mulai dari 6
Part 1 — loop retry dengan history semakin pendek:


        while history_limit >= MIN_HISTORY_LIMIT:
            try:
                context_dict, json_flag = self.prepare_context(trace, history_limit)
                # ^ siapkan konteks dengan history_limit saat ini

                system_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
                    .render(
                        targets=self.targets,                    # "factors"
                        scenario=self.scen.get_scenario_all_desc(
                            filtered_tag="hypothesis_and_experiment"
                        ),
                        # ^ deskripsi skenario lengkap (dari QlibAlphaAgentScenario)
                        hypothesis_output_format=context_dict["hypothesis_output_format"],
                        hypothesis_specification=context_dict["hypothesis_specification"],
                    )
                )
                # ^ system prompt: "You are an expert quant researcher.
                #   You are working with Qlib... Data available: ...
                #   Output format: JSON with fields: hypothesis, concise_observation, ..."

                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
                    .render(
                        targets=self.targets,
                        hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                        # ^ riwayat percobaan sebelumnya ATAU direction awal
                        RAG=context_dict["RAG"],
                        round=len(trace.hist)
                        # ^ nomor round saat ini (0, 1, 2, ...)
                        #   bisa dipakai di template: "This is round {{round}}..."
                    )
                )
Part 2 — kirim ke LLM dan parse:


                resp = self._call_llm(user_prompt, system_prompt, json_flag)
                # ^ KIRIM KE LLM! Return string JSON

                hypothesis = self.convert_response(resp)
                # ^ parse JSON → AlphaAgentHypothesis

                return hypothesis
                # ^ sukses → langsung return
Part 3 — error handling: input terlalu panjang:


            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                    # ^ prompt terlalu panjang!
                    #   kurangi history_limit: 6→5→4→3→2→1
                    #   riwayat yang dikirim makin sedikit → prompt makin pendek
                else:
                    raise
                    # ^ bukan error token limit → re-raise (biarkan caller handle)
Part 4 — last resort:


        # Keluar while loop (history_limit < MIN_HISTORY_LIMIT)
        # Coba terakhir kali dengan history_limit=1
        context_dict, json_flag = self.prepare_context(trace, MIN_HISTORY_LIMIT)
        # ... render prompt yang sama ...
        resp = self._call_llm(user_prompt, system_prompt, json_flag)
        hypothesis = self.convert_response(resp)
        return hypothesis
        # ^ kalau ini masih gagal → exception naik ke caller
        #   LoopBase.run() akan menangkap sebagai skip_loop_error jika FactorEmptyError
BAGIAN 5: class AlphaAgentHypothesis2FactorExpression (baris 350–634) — YANG DIPAKAI DI PIPELINE
Di-load via import_class("factors.proposal.AlphaAgentHypothesis2FactorExpression"). Ini yang mengubah hipotesis menjadi ekspresi faktor matematika.


class AlphaAgentHypothesis2FactorExpression(FactorHypothesis2Experiment):
    # ^ inherit: Hypothesis2Experiment → LLMHypothesis2Experiment
    #            → FactorHypothesis2Experiment → ini

    def __init__(self, *args, consistency_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)    # set targets="factors"

        from factors.coder.config import FACTOR_COSTEER_
Grep "class FactorRegulator" (in /root/projects/first-experiment/ai-agent/backend)
2 lines of output
Read factor_regulator.py
Baik, semua file yang dibutuhkan sudah terbaca. Mari mulai.

FILE: factors/proposal.py — Proposal classes: LLM generate hypothesis & faktor
File ini berisi class-class yang menghubungkan LLM dengan pipeline — mereka yang menyiapkan prompt, memanggil LLM, dan parse response.

BAGIAN 1: Import (baris 1–18)

import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined
# ^ jinja2: template engine
#   dipakai untuk render prompt YAML → string final yang dikirim ke LLM
#   StrictUndefined: kalau variable di template tidak ada → raise error (bukan silent)

from factors.coder.factor import FactorExperiment, FactorTask
# ^ FactorTask: dataclass satu faktor (name, description, formulation, expression, variables)
#   FactorExperiment: alias dari Experiment (FactorExperiment = Experiment)

from components.proposal import FactorHypothesisGen, FactorHypothesis2Experiment
# ^ parent class dari proposal classes — kita telusuri setelah ini

from core.prompts import Prompts
# ^ class yang load YAML file → dict-like object

from core.proposal import Hypothesis, Scenario, Trace
# ^ sudah kita bahas: Hypothesis (ide trading), Scenario (deskripsi market), Trace (riwayat)

from core.experiment import Experiment
from factors.experiment import QlibFactorExperiment
# ^ subclass Experiment untuk Qlib

from llm.client import LocalLLMBackend, robust_json_parse
# ^ LocalLLMBackend: panggil LLM
#   robust_json_parse: parse JSON dari response LLM (toleran terhadap format jelek)

import os
import pandas as pd
from log import logger
from factors.regulator.factor_regulator import FactorRegulator
# ^ FactorRegulator: cek apakah ekspresi faktor valid, tidak duplikat, tidak terlalu kompleks
Sekarang FactorHypothesisGen dan FactorHypothesis2Experiment di-import — mari telusuri.

MASUK KE components/proposal/init.py — Parent class
class LLMHypothesisGen(HypothesisGen) (baris 21–58) — base class: generate hypothesis via LLM

class LLMHypothesisGen(HypothesisGen):
    # ^ inherit dari core.proposal.HypothesisGen (ABC dengan method gen())

    def __init__(self, scen: Scenario):
        super().__init__(scen)    # simpan self.scen

    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...
    # ^ abstract: subclass harus menyiapkan dict context untuk template prompt
    #   return (context_dict, json_flag)
    #   json_flag = True → LLM diminta output JSON

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...
    # ^ abstract: parse response string dari LLM → Hypothesis object
Method gen() — implementasi default:


    def gen(self, trace: Trace) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)
        # ^ subclass menyiapkan: hypothesis_and_feedback, RAG, output_format, specification

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                targets=self.targets,
                # ^ "factors" — diset di FactorHypothesisGen
                scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                # ^ deskripsi lengkap scenario (market, data, interface)
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        # ^ render template system prompt dari YAML
        #   template berisi placeholder: {{ targets }}, {{ scenario }}, dll
        #   jinja2 mengganti placeholder dengan nilai aktual

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=self.targets,
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                # ^ riwayat hipotesis+feedback sebelumnya (atau "first round" message)
                RAG=context_dict["RAG"],
                # ^ Retrieval-Augmented Generation context (None untuk sekarang)
            )
        )

        resp = LocalLLMBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag
        )
        # ^ panggil LLM: kirim system prompt + user prompt → dapat response string

        hypothesis = self.convert_response(resp)
        # ^ parse JSON response → Hypothesis object
        return hypothesis
Note: prompt_dict di sini diambil dari components/proposal/prompts.yaml — bukan dari factors/prompts/. Ini prompt default. AlphaAgentHypothesisGen nanti meng-override gen() dan pakai prompt sendiri dari factors/prompts/prompts.yaml.

class FactorHypothesisGen(LLMHypothesisGen) (baris 61–64)

class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"
        # ^ ini yang dipakai di template: {{ targets }}
        #   mengarahkan LLM bahwa kita sedang generate FAKTOR (bukan model)
class LLMHypothesis2Experiment(Hypothesis2Experiment) (baris 79–110)

class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, trace: Trace) -> Experiment: ...

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                experiment_output_format=context["experiment_output_format"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                target_list=context["target_list"],
                RAG=context["RAG"],
            )
        )

        resp = LocalLLMBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag
        )
        return self.convert_response(resp, trace)
Sama polanya: prepare context → render prompt → panggil LLM → parse response.

class FactorHypothesis2Experiment(LLMHypothesis2Experiment) (baris 113–116)

class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"
Sekarang Prompts juga dipakai — sekilas.

Sekilas core/prompts.py:


class Prompts(SingletonBaseClass, dict[str, str]):
    # ^ inherit dari dict DAN SingletonBaseClass
    #   artinya: Prompts bertindak seperti dict biasa
    #   DAN Singleton: instance dengan file_path yang sama hanya dibuat sekali

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        with file_path.open(encoding="utf8") as file:
            prompt_yaml_dict = yaml.safe_load(file)
            # ^ baca file YAML → dict Python

        for key, value in prompt_yaml_dict.items():
            self[key] = value
            # ^ copy semua entry ke dict ini
            #   setelah ini: prompts["hypothesis_gen"] → template string
Kembali ke factors/proposal.py

BAGIAN 2: Utility functions (baris 19–54)

DEFAULT_HISTORY_LIMIT = 6      # default: kirim 6 riwayat terakhir ke LLM
MIN_HISTORY_LIMIT = 1          # minimum: paling sedikit 1 riwayat
render_hypothesis_and_feedback() (baris 23–34)

def render_hypothesis_and_feedback(prompt_dict, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> str:
    if len(trace.hist) > 0:
        limited_trace = Trace(scen=trace.scen)
        # ^ buat Trace baru dengan riwayat terbatas

        limited_trace.hist = trace.hist[-history_limit:] if history_limit > 0 else trace.hist
        # ^ ambil N riwayat terakhir saja
        #   misal history_limit=6, trace.hist punya 20 item → ambil 14–19

        return (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=limited_trace)
        )
        # ^ render template "hypothesis_and_feedback" dari YAML
        #   template ini mengiterasi trace.hist dan menampilkan:
        #   - hipotesis sebelumnya
        #   - feedback dari evaluator
        #   → menjadi konteks yang dikirim ke LLM supaya tahu apa yang sudah dicoba
    else:
        return "No previous hypothesis and feedback available since it's the first round."
        # ^ loop pertama: belum ada riwayat
is_input_length_error() (baris 37–50)

def is_input_length_error(error_msg: str) -> bool:
    error_indicators = [
        "input length", "context length", "maximum context",
        "token limit", "InvalidParameter", "Range of input length",
        "max_tokens", "too long"
    ]
    error_str = str(error_msg).lower()
    return any(indicator.lower() in error_str for indicator in error_indicators)
    # ^ cek apakah error message mengandung kata kunci terkait panjang input
    #   kalau True → berarti prompt terlalu panjang → perlu kurangi riwayat

QlibFactorHypothesis = Hypothesis
# ^ alias: QlibFactorHypothesis sama saja dengan Hypothesis standar

qa_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "proposal.yaml")
# ^ load PERTAMA: dari factors/prompts/proposal.yaml
#   (nanti di baris 199 di-reassign ke prompts.yaml — ini mungkin bug/legacy)
BAGIAN 3: class AlphaAgentHypothesis(Hypothesis) (baris 56–86)

class AlphaAgentHypothesis(Hypothesis):
    # ^ extends Hypothesis: menambahkan concise_specification

    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str       # BARU: spesifikasi ringkas
    ) -> None:
        super().__init__(
            hypothesis,
            "",                          # reason → kosong (tidak dipakai AlphaAgent)
            "",                          # concise_reason → kosong
            concise_observation,
            concise_justification,
            concise_knowledge,
        )
        self.concise_specification = concise_specification
        # ^ spesifikasi teknis: bagaimana hipotesis ini akan diimplementasikan
        #   misal: "Use 20-day rolling window, normalize by ATR"

    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                concise Specification: {self.concise_specification}
                """
        # ^ representasi string yang dikirim ke LLM di step berikutnya

base_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")
# ^ load prompt dari factors/prompts/prompts.yaml
#   ini dipakai oleh QlibFactorHypothesisGen (bukan AlphaAgent)
BAGIAN 4: class QlibFactorHypothesisGen(FactorHypothesisGen) (baris 90–122)
Ini versi standar/basic — bukan AlphaAgent.


class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)      # self.targets = "factors"

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(base_prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        # ^ sama seperti render_hypothesis_and_feedback tapi tanpa history limit

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,                 # tidak ada RAG
            "hypothesis_output_format": base_prompt_dict["hypothesis_output_format"],
            # ^ template JSON format yang LLM harus ikuti
            "hypothesis_specification": base_prompt_dict["factor_hypothesis_specification"],
            # ^ instruksi spesifik untuk generate hypothesis faktor
        }
        return context_dict, True        # True = minta JSON output

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = robust_json_parse(response)
        # ^ parse JSON dari LLM (toleran terhadap trailing comma, dll)

        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            reason=response_dict.get("reason", ""),
            concise_reason=response_dict.get("concise_reason", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
        )
        # ^ buat Hypothesis dari field JSON
        return hypothesis
BAGIAN 5: class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment) (baris 125–195)
Versi standar: convert hypothesis → faktor.


class QlibFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        # ^ deskripsi lengkap scenario

        experiment_output_format = base_prompt_dict["factor_experiment_output_format"]
        # ^ format JSON yang LLM harus ikuti untuk generate faktor

        hypothesis_and_feedback = (
            Environment(undefined=StrictUndefined)
            .from_string(base_prompt_dict["hypothesis_and_feedback"])
            .render(trace=trace)
        ) if len(trace.hist) > 0
        else "No previous hypothesis and feedback available since it's the first round."

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]
        # ^ t = (hypothesis, experiment, feedback)
        #   t[1] = experiment dari setiap loop sebelumnya

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)
            # ^ kumpulkan SEMUA faktor dari semua eksperimen sebelumnya
            #   ini dikirim ke LLM supaya tahu faktor apa yang sudah dibuat

        return {
            "target_hypothesis": str(hypothesis),     # hipotesis saat ini (string)
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,               # faktor-faktor sebelumnya
            "RAG": None,
        }, True

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = robust_json_parse(response)
        tasks = []

        for factor_name in response_dict:
            factor_data = response_dict.get(factor_name, {})
            if not isinstance(factor_data, dict):
                continue                    # skip entry yang bukan dict

            description = factor_data.get("description", "")
            formulation = factor_data.get("formulation", "")
            variables = factor_data.get("variables", {})

            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    variables=variables,
                )
            )
Sekarang FactorTask dipakai — mari sekilas.

Sekilas factors/coder/factor.py — FactorTask:


class FactorTask(CoSTEERTask):
    def __init__(self, factor_name, factor_description, factor_formulation,
                 factor_expression=None, variables={}, resource=None,
                 factor_implementation=False):
        self.factor_name = factor_name               # "vol_mom_ratio"
        self.factor_description = factor_description  # "Ratio of volatility to momentum"
        self.factor_formulation = factor_formulation  # rumus matematika informal
        self.factor_expression = factor_expression    # rumus formal: "Std(Close,20)/Mean(Volume,10)"
        self.variables = variables                    # {"Close": "closing price", "Volume": "..."}
        self.factor_resources = resource
        self.factor_implementation = factor_implementation  # False = belum diimplementasi
        super().__init__(name=factor_name)

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""
        # ^ representasi string yang disisipkan ke prompt
Kembali ke convert_response():


        exp = QlibFactorExperiment(tasks)
        # ^ buat Experiment dengan list FactorTask

        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]
        # ^ based_experiments: experiment-experiment sebelumnya yang berhasil (feedback=True)
        #   ditambah satu experiment kosong di depan (baseline)

        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)
        # ^ DEDUPLIKASI: buang faktor yang namanya sama dengan yang sudah ada
        #   di experiment sebelumnya — jangan buat faktor yang sudah pernah dibuat

        exp.tasks = unique_tasks
        return exp
BAGIAN 6: class AlphaAgentHypothesisGen(FactorHypothesisGen) (baris 202–321)
Ini yang dipakai di AlphaAgentLoop — versi utama dengan direction support dan history limit retry.


qa_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")
# ^ REASSIGN: qa_prompt_dict sekarang dari prompts.yaml (bukan proposal.yaml)
#   ini yang sebenarnya dipakai oleh AlphaAgent classes

class AlphaAgentHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario, potential_direction: str=None):
        super().__init__(scen)            # self.targets = "factors"
        self.potential_direction = potential_direction
        # ^ direction dari user/planning, misal "momentum cross-sectional"
        #   bisa juga berisi strategy_suffix + external_context (effective_direction)
Method prepare_context() (baris 207–229) — siapkan context untuk prompt

    def prepare_context(self, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict, bool]:

        if len(trace.hist) > 0:
            hypothesis_and_feedback = render_hypothesis_and_feedback(
                qa_prompt_dict, trace, history_limit
            )
            # ^ ada riwayat → render template dengan N riwayat terakhir

        elif self.potential_direction is not None:
            hypothesis_and_feedback = (
                Environment(undefined=StrictUndefined)
                .from_string(qa_prompt_dict["potential_direction_transformation"])
                .render(potential_direction=self.potential_direction)
            )
            # ^ LOOP PERTAMA + ADA DIRECTION:
            #   render template "potential_direction_transformation"
            #   template ini mengubah direction menjadi konteks awal
            #   misal: "Your starting point for exploration is: momentum cross-sectional.
            #           Consider factors related to this direction..."

        else:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round. You are encouraged to propose an innovative hypothesis that diverges significantly from existing perspectives."
            # ^ LOOP PERTAMA + TANPA DIRECTION:
            #   LLM bebas eksplorasi, didorong untuk inovatif

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": qa_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": qa_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True
        # ^ True = minta JSON output dari LLM
Method convert_response() (baris 231–244) — parse LLM JSON → AlphaAgentHypothesis

    def convert_response(self, response: str) -> AlphaAgentHypothesis:
        response_dict = robust_json_parse(response)

        hypothesis = AlphaAgentHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_specification=response_dict.get("concise_specification", ""),
        )
        # ^ pakai .get() dengan default "" → tidak crash kalau field tidak ada
        return hypothesis
Method _call_llm() (baris 246–256)

    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        return self.llm_backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt, system_prompt=system_prompt,
            json_mode=json_mode,
        )
        # ^ self.llm_backend di-inject oleh AlphaAgentLoop.__init__()
        #   baris: self.hypothesis_generator.llm_backend = llm_backend
        #   panggil LocalLLMBackend.build_messages_and_create_chat_completion()
        #   → kirim prompt ke model → dapat response string
Method gen() (baris 258–321) — override gen() dari parent dengan retry logic
Ini yang benar-benar dipanggil di AlphaAgentLoop.factor_propose().


    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        history_limit = DEFAULT_HISTORY_LIMIT    # mulai dari 6
Part 1 — retry loop:


        while history_limit >= MIN_HISTORY_LIMIT:
            # ^ coba dengan history_limit 6, 5, 4, 3, 2, 1
            try:
                context_dict, json_flag = self.prepare_context(trace, history_limit)
                # ^ siapkan context dengan history_limit saat ini
Part 2 — render prompt:


                system_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
                    .render(
                        targets=self.targets,                    # "factors"
                        scenario=self.scen.get_scenario_all_desc(
                            filtered_tag="hypothesis_and_experiment"
                        ),
                        # ^ deskripsi scenario yang difilter untuk hypothesis+experiment

                        hypothesis_output_format=context_dict["hypothesis_output_format"],
                        # ^ JSON format yang harus diikuti LLM

                        hypothesis_specification=context_dict["hypothesis_specification"],
                        # ^ instruksi spesifik tentang cara generate hypothesis
                    )
                )

                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
                    .render(
                        targets=self.targets,
                        hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                        # ^ riwayat / direction context
                        RAG=context_dict["RAG"],
                        round=len(trace.hist)
                        # ^ nomor round saat ini
                    )
                )
Part 3 — panggil LLM dan parse:


                resp = self._call_llm(user_prompt, system_prompt, json_flag)
                # ^ kirim ke LLM → dapat string response (JSON)

                hypothesis = self.convert_response(resp)
                # ^ parse JSON → AlphaAgentHypothesis

                return hypothesis
                # ^ sukses → langsung return
Part 4 — handle error:


            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                    # ^ prompt terlalu panjang → kurangi riwayat, coba lagi
                    #   6→5→4→3→2→1 sampai muat di context window LLM
                else:
                    raise
                    # ^ error lain (bukan input length) → langsung raise
Part 5 — last attempt:


        # Kalau while loop selesai tanpa return (artinya semua retry gagal)
        context_dict, json_flag = self.prepare_context(trace, MIN_HISTORY_LIMIT)
        # ... render prompt dengan history_limit=1 ...
        resp = self._call_llm(user_prompt, system_prompt, json_flag)
        hypothesis = self.convert_response(resp)
        return hypothesis
        # ^ percobaan terakhir tanpa try/except → kalau gagal → crash
BAGIAN 7: class EmptyHypothesisGen(FactorHypothesisGen) (baris 325–345)

class EmptyHypothesisGen(FactorHypothesisGen):
    # ^ dipakai oleh BacktestLoop (FactorBackTestBasePropSetting)
    #   "EmptyHypothesisGen" karena backtest tidak perlu generate hypothesis baru

    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        hypothesis = AlphaAgentHypothesis(
            hypothesis="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
            concise_specification=""
        )
        return hypothesis
        # ^ return hypothesis kosong — backtest langsung ke step construct
BAGIAN 8: class AlphaAgentHypothesis2FactorExpression(FactorHypothesis2Experiment) (baris 350–634)
Ini yang PALING penting dan paling kompleks — convert hypothesis → faktor + validasi.


class AlphaAgentHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, *args, consistency_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # ^ self.targets = "factors"
Oke, aku punya semua file yang dibutuhkan. Mari mulai.

FILE: factors/proposal.py — Class-class yang membuat hipotesis dan mengubahnya jadi faktor
File ini berisi implementasi konkret dari ABC yang ada di core/proposal.py — ini tempat LLM benar-benar dipanggil.

BAGIAN 1: Import (baris 1–18)

import json
from pathlib import Path
from typing import List, Tuple
from jinja2 import Environment, StrictUndefined
# ^ Jinja2: template engine. StrictUndefined = error kalau variabel template tidak ada

from factors.coder.factor import FactorExperiment, FactorTask
# ^ FactorTask: dataclass satu faktor (nama, deskripsi, rumus, ekspresi)
#   FactorExperiment: alias dari Experiment (di factor.py baris 246: FactorExperiment = Experiment)

from components.proposal import FactorHypothesisGen, FactorHypothesis2Experiment
# ^ parent class yang menyediakan base flow: prepare_context → call LLM → convert_response
Sebelum lanjut, kita perlu memahami parent class ini.

Pindah ke components/proposal/init.py — FactorHypothesisGen

class LLMHypothesisGen(HypothesisGen) (baris 21–58)

class LLMHypothesisGen(HypothesisGen):
    # ^ inherit dari core.proposal.HypothesisGen (ABC)

    def __init__(self, scen: Scenario):
        super().__init__(scen)

    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...
    # ^ subclass harus implement: siapkan context dict + flag json_mode

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...
    # ^ subclass harus implement: parse response LLM → Hypothesis object

    def gen(self, trace: Trace) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)
        # ^ panggil subclass untuk menyiapkan konteks

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                targets=self.targets,                    # "factors"
                scenario=self.scen.get_scenario_all_desc(
                    filtered_tag="hypothesis_and_experiment"
                ),                                        # deskripsi lengkap skenario
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        # ^ render template Jinja2 dari YAML prompt file
        #   system_prompt berisi: instruksi LLM + scenario + format output

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=self.targets,
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                RAG=context_dict["RAG"],
            )
        )
        # ^ user_prompt berisi: riwayat hipotesis+feedback sebelumnya + RAG (null)

        resp = LocalLLMBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag
        )
        # ^ PANGGIL LLM! kirim system_prompt + user_prompt → LLM balas JSON

        hypothesis = self.convert_response(resp)
        # ^ parse JSON response → Hypothesis object
        return hypothesis
Ini adalah base flow yang di-override oleh AlphaAgentHypothesisGen di factors/proposal.py.

class FactorHypothesisGen(LLMHypothesisGen) (baris 61–64)

class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"
        # ^ hanya set targets = "factors"
        #   ini dipakai di template prompt: "You are generating {targets}..."
class LLMHypothesis2Experiment(Hypothesis2Experiment) (baris 79–110)

class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, trace: Trace) -> Experiment: ...

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.get_scenario_all_desc(...),
                experiment_output_format=context["experiment_output_format"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                target_list=context["target_list"],
                RAG=context["RAG"],
            )
        )

        resp = LocalLLMBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag
        )
        return self.convert_response(resp, trace)
        # ^ flow yang sama: prepare context → render prompt → call LLM → parse response
class FactorHypothesis2Experiment(LLMHypothesis2Experiment) (baris 113–116)

class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"    # set targets
Sekarang juga Prompts:

Sekilas core/prompts.py — Prompts:


class Prompts(SingletonBaseClass, dict[str, str]):
    # ^ Singleton + dict: satu file YAML = satu instance, bisa diakses prompt_dict["key"]

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        with file_path.open(encoding="utf8") as file:
            prompt_yaml_dict = yaml.safe_load(file)
            # ^ parse YAML file → Python dict

        for key, value in prompt_yaml_dict.items():
            self[key] = value
            # ^ setiap key di YAML jadi key di dict ini
            #   misal: prompt_dict["hypothesis_gen"]["system_prompt"] = "You are..."
Kembali ke factors/proposal.py — lanjut import


from core.prompts import Prompts
from core.proposal import Hypothesis, Scenario, Trace
from core.experiment import Experiment
from factors.experiment import QlibFactorExperiment
# ^ Experiment subclass khusus Qlib: override workspace dengan QlibFBWorkspace

from llm.client import LocalLLMBackend, robust_json_parse
# ^ LocalLLMBackend: class untuk memanggil LLM (local model via vLLM/transformers)
#   robust_json_parse: parse JSON yang mungkin ada formatting error

import os
import pandas as pd
from log import logger
from factors.regulator.factor_regulator import FactorRegulator
# ^ evaluator: cek duplikasi, kompleksitas, dan validitas ekspresi faktor
BAGIAN 2: Helper functions (baris 19–54)

DEFAULT_HISTORY_LIMIT = 6     # max riwayat yang dikirim ke LLM
MIN_HISTORY_LIMIT = 1         # minimum riwayat (1 entry)
render_hypothesis_and_feedback() (baris 23–34)

def render_hypothesis_and_feedback(prompt_dict, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> str:
    if len(trace.hist) > 0:
        limited_trace = Trace(scen=trace.scen)
        # ^ buat Trace baru kosong (dengan scenario yang sama)

        limited_trace.hist = trace.hist[-history_limit:] if history_limit > 0 else trace.hist
        # ^ ambil hanya N entry terakhir dari riwayat
        #   kalau trace.hist punya 10 entry dan history_limit=6,
        #   ambil entry ke-4 sampai 9 (6 terakhir)
        #   ini mencegah prompt terlalu panjang

        return (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=limited_trace)
        )
        # ^ render template Jinja2 "hypothesis_and_feedback" dengan trace terbatas
        #   template ini iterate trace.hist dan format setiap entry jadi teks:
        #   "Round 1: Hypothesis: ... Feedback: ... Decision: ..."

    else:
        return "No previous hypothesis and feedback available since it's the first round."
        # ^ round pertama: belum ada riwayat
is_input_length_error() (baris 37–50)

def is_input_length_error(error_msg: str) -> bool:
    error_indicators = [
        "input length", "context length", "maximum context",
        "token limit", "InvalidParameter", "Range of input length",
        "max_tokens", "too long"
    ]
    error_str = str(error_msg).lower()
    return any(indicator.lower() in error_str for indicator in error_indicators)
    # ^ cek apakah error message berisi kata-kata yang menunjukkan
    #   prompt terlalu panjang untuk LLM
    #   dipakai untuk strategi retry: kurangi riwayat, coba lagi
BAGIAN 3: Prompt loading (baris 53–88)

QlibFactorHypothesis = Hypothesis
# ^ alias: QlibFactorHypothesis = Hypothesis dari core/proposal.py
#   dipakai oleh QlibFactorHypothesisGen

qa_prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts" / "proposal.yaml")
# ^ load YAML prompt file khusus AlphaAgent
#   path: factors/prompts/proposal.yaml
#   berisi template prompt untuk hypothesis generation dan factor conversion
BAGIAN 4: class AlphaAgentHypothesis(Hypothesis) (baris 56–86) — Hypothesis khusus AlphaAgent

class AlphaAgentHypothesis(Hypothesis):
    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str      # FIELD BARU: tidak ada di parent Hypothesis
    ) -> None:
        super().__init__(
            hypothesis,
            "",                          # reason: dikosongkan (tidak dipakai)
            "",                          # concise_reason: dikosongkan
            concise_observation,
            concise_justification,
            concise_knowledge,
        )
        self.concise_specification = concise_specification
        # ^ field tambahan: spesifikasi teknis yang lebih detail
        #   misal: "Use 20-day rolling window for volatility calculation"

    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                concise Specification: {self.concise_specification}
                """
        # ^ format output yang dikirim ke step selanjutnya (factor_construct)
BAGIAN 5: class QlibFactorHypothesisGen (baris 90–122) — versi "basic" Qlib
Class ini dipakai oleh FactorBasePropSetting (mode tradisional), BUKAN oleh AlphaAgent. Jelaskan singkat.


class QlibFactorHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)       # self.targets = "factors"

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        # ^ render riwayat + format output → context dict
        hypothesis_and_feedback = (
            Environment(undefined=StrictUndefined)
            .from_string(base_prompt_dict["hypothesis_and_feedback"])
            .render(trace=trace)
        ) if len(trace.hist) > 0
        else "No previous hypothesis..."

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": base_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": base_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True     # True = json_mode

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = robust_json_parse(response)
        # ^ parse JSON string → dict

        hypothesis = QlibFactorHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            reason=response_dict.get("reason", ""),
            concise_reason=response_dict.get("concise_reason", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
        )
        return hypothesis
        # ^ QlibFactorHypothesis = Hypothesis (alias di baris 53)
BAGIAN 6: class AlphaAgentHypothesisGen (baris 202–321) — class utama untuk generate hipotesis
Ini yang dipakai oleh ALPHA_AGENT_FACTOR_PROP_SETTING → dipanggil di AlphaAgentLoop.factor_propose().


class AlphaAgentHypothesisGen(FactorHypothesisGen):
    # ^ inherit: FactorHypothesisGen → LLMHypothesisGen → HypothesisGen
    #   self.targets = "factors" (dari FactorHypothesisGen)

    def __init__(self, scen: Scenario, potential_direction: str = None):
        super().__init__(scen)
        self.potential_direction = potential_direction
        # ^ direction yang sudah dibangun di AlphaAgentLoop.__init__:
        #   base direction + strategy_suffix + external_context
Method prepare_context() (baris 207–229) — siapkan input untuk LLM

    def prepare_context(self, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict, bool]:

        if len(trace.hist) > 0:
            hypothesis_and_feedback = render_hypothesis_and_feedback(
                qa_prompt_dict, trace, history_limit
            )
            # ^ ada riwayat: render N entry terakhir jadi teks
            #   berisi: hipotesis sebelumnya, backtest result, feedback, decision

        elif self.potential_direction is not None:
            hypothesis_and_feedback = (
                Environment(undefined=StrictUndefined)
                .from_string(qa_prompt_dict["potential_direction_transformation"])
                .render(potential_direction=self.potential_direction)
            )
            # ^ round pertama + ada direction: render template khusus
            #   mengubah direction jadi format yang bisa dipahami LLM
            #   misal template: "You are given the following research direction:
            #   {potential_direction}. Use this as starting point..."

        else:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round. You are encouraged to propose an innovative hypothesis that diverges significantly from existing perspectives."
            # ^ round pertama + tanpa direction: LLM explore bebas

        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,                                    # Retrieval-Augmented Generation: tidak aktif
            "hypothesis_output_format": qa_prompt_dict["hypothesis_output_format"],
            # ^ format JSON yang diharapkan:
            #   {"hypothesis": "...", "concise_observation": "...", ...}
            "hypothesis_specification": qa_prompt_dict["factor_hypothesis_specification"],
            # ^ instruksi spesifik: apa yang harus ada di hipotesis
        }
        return context_dict, True    # True = json_mode (LLM harus output JSON)
Method convert_response() (baris 231–244) — parse JSON → AlphaAgentHypothesis

    def convert_response(self, response: str) -> AlphaAgentHypothesis:
        response_dict = robust_json_parse(response)
        # ^ parse JSON, toleran terhadap formatting error
        #   misal: extra comma, missing quotes → tetap bisa parse

        hypothesis = AlphaAgentHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_specification=response_dict.get("concise_specification", ""),
        )
        # ^ .get("key", "") → default empty string kalau key tidak ada
        #   mencegah KeyError kalau LLM lupa generate salah satu field
        return hypothesis
Method _call_llm() (baris 246–256) — wrapper untuk panggil LLM

    def _call_llm(self, user_prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        return self.llm_backend.build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=json_mode,
        )
        # ^ self.llm_backend di-set oleh AlphaAgentLoop.__init__:
        #   self.hypothesis_generator.llm_backend = llm_backend
        #   method ini dipanggil oleh gen() di bawah
        #
        #   build_messages_and_create_chat_completion():
        #   1. format messages: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        #   2. kirim ke model (local vLLM atau API)
        #   3. return string response
Method gen() (baris 258–321) — OVERRIDE dari parent — generate hipotesis dengan retry

    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        history_limit = DEFAULT_HISTORY_LIMIT    # mulai dengan 6

        while history_limit >= MIN_HISTORY_LIMIT:
            # ^ loop: coba generate, kalau prompt terlalu panjang → kurangi riwayat

            try:
                context_dict, json_flag = self.prepare_context(trace, history_limit)
                # ^ siapkan konteks dengan history_limit saat ini
Part — render prompt:


                system_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
                    .render(
                        targets=self.targets,              # "factors"
                        scenario=self.scen.get_scenario_all_desc(
                            filtered_tag="hypothesis_and_experiment"
                        ),                                  # scenario description
                        hypothesis_output_format=context_dict["hypothesis_output_format"],
                        hypothesis_specification=context_dict["hypothesis_specification"],
                    )
                )
                # ^ system prompt: "You are a quantitative researcher.
                #   Given the following market scenario: {scenario}
                #   Generate a hypothesis for {targets}.
                #   Output format: {hypothesis_output_format}
                #   Specification: {hypothesis_specification}"

                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
                    .render(
                        targets=self.targets,
                        hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                        # ^ riwayat sebelumnya ATAU direction ATAU "first round"
                        RAG=context_dict["RAG"],     # None
                        round=len(trace.hist)         # nomor round saat ini
                    )
                )
                # ^ user prompt: "Here is the history: {hypothesis_and_feedback}
                #   This is round {round}. Propose a new hypothesis."
Part — panggil LLM dan parse:


                resp = self._call_llm(user_prompt, system_prompt, json_flag)
                # ^ KIRIM KE LLM! return JSON string

                hypothesis = self.convert_response(resp)
                # ^ parse JSON → AlphaAgentHypothesis

                return hypothesis
                # ^ sukses! langsung return
Part — error handling (retry dengan riwayat lebih sedikit):


            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                    # ^ prompt terlalu panjang → kurangi riwayat 1 entry
                    #   loop kembali ke while, coba lagi
                    #   6 → 5 → 4 → 3 → 2 → 1
                else:
                    raise
                    # ^ error bukan karena panjang prompt → raise (serahkan ke caller)
Part — fallback terakhir (baris 297–321):


        # ^ keluar while loop: history_limit < MIN_HISTORY_LIMIT
        #   tapi belum return → coba sekali lagi dengan MIN_HISTORY_LIMIT
        context_dict, json_flag = self.prepare_context(trace, MIN_HISTORY_LIMIT)
        # ... render prompt yang sama ...
        resp = self._call_llm(user_prompt, system_prompt, json_flag)
        hypothesis = self.convert_response(resp)
        return hypothesis
        # ^ percobaan terakhir: kalau masih gagal → exception naik ke atas
BAGIAN 7: class EmptyHypothesisGen (baris 325–345) — placeholder kosong

class EmptyHypothesisGen(FactorHypothesisGen):
    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        hypothesis = AlphaAgentHypothesis(
            hypothesis="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
            concise_specification=""
        )
        return hypothesis
        # ^ TIDAK panggil LLM — return hipotesis kosong
        #   dipakai oleh BacktestLoop: dia sudah punya faktor dari file,
        #   tidak perlu generate hipotesis baru
BAGIAN 8: class AlphaAgentHypothesis2FactorExpression (baris 350–634) — class terbesar: convert hipotesis → faktor
Ini yang dipanggil di AlphaAgentLoop.factor_construct().


class AlphaAgentHypothesis2FactorExpression(FactorHypothesis2Experiment):
    # ^ inherit: FactorHypothesis2Experiment → LLMHypothesis2Experiment → Hypothesis2Experiment

    def __init__(self, *args, consistency_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # ^ self.targets = "factors" (dari FactorHypothesis2Experiment)

        from factors.coder.config import FACTOR_COSTEER_SETTINGS
        self.factor_regulator = FactorRegulator(
            factor_zoo_path=FACTOR_COSTEER_SETTINGS.factor_zoo_path,
            duplication_threshold=FACTOR_COSTEER_SETTINGS.duplication_threshold
        )
        # ^ FactorRegulator: mengevaluasi ekspresi faktor
        #   - apakah bisa di-parse?
        #   - apakah terlalu mirip dengan faktor yang sudah ada (alpha zoo)?
        #   - apakah terlalu kompleks?
Sekilas factors/regulator/factor_regulator.py — FactorRegulator:


class FactorRegulator(Evaluator):
    def __init__(self, factor_zoo_path=None, duplication_threshold=8,
                 symbol_length_threshold=300, base_features_threshold=6):
        self.factor_zoo_path = factor_zoo_path
        if factor_zoo_path:
            self.alphazoo = pd.read_csv(factor_zoo_path, index_col=None)
            # ^ load CSV berisi semua faktor yang sudah diketahui ("alpha zoo")
            #   dipakai untuk cek apakah faktor baru terlalu mirip
        else:
            self.alphazoo = pd.DataFrame()

        self.duplication_threshold = duplication_threshold
        # ^ threshold: kalau subtree yang sama > 8 nodes → dianggap duplikat

        self.symbol_length_threshold = symbol_length_threshold
        # ^ max panjang simbol ekspresi (300) → terlalu panjang = terlalu kompleks

        self.base_features_threshold = base_features_threshold
        # ^ max jumlah fitur dasar unik (6) → terlalu banyak = over-fitted

        self.new_factors = []
        # ^ buffer faktor baru yang ditambahkan dalam run ini

    def is_parsable(self, expression: str) -> bool:
        try:
            parse_expression(expression)     # AST parser untuk ekspresi faktor
            return True
        except Exception:
            return False
Kembali ke factors/proposal.py


        self.consistency_enabled = consistency_enabled
        self._quality_gate = None
        # ^ lazy-loaded quality gate

    @property
    def quality_gate(self):
        if self._quality_gate is None and self.consistency_enabled:
            try:
                from factors.regulator.consistency_checker import FactorQualityGate
                self._quality_gate = FactorQualityGate(
                    consistency_enabled=self.consistency_enabled,
                    complexity_enabled=True,
                    redundancy_enabled=True
                )
            except ImportError as e:
                logger.warning(f"Could not load consistency checker: {e}")
                self._quality_gate = None
        return self._quality_gate
        # ^ lazy-load: hanya buat kalau dibutuhkan DAN consistency_enabled=True
        #   FactorQualityGate: cek apakah faktor konsisten dengan hipotesis
Method prepare_context() (baris 380–402) — siapkan konteks untuk LLM

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace,
                        history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict, bool]:
        scenario = trace.scen.get_scenario_all_desc()
        # ^ deskripsi lengkap skenario Qlib

        experiment_output_format = qa_prompt_dict["factor_experiment_output_format"]
        # ^ format JSON yang diharapkan:
        #   {"factor_name_1": {"description": "...", "formulation": "...",
        #    "expression": "...", "variables": {...}}, ...}

        function_lib_description = qa_prompt_dict['function_lib_description']
        # ^ deskripsi library fungsi yang bisa dipakai di ekspresi
        #   misal: "Ts_Mean(x, d): rolling mean of x over d days"

        hypothesis_and_feedback = render_hypothesis_and_feedback(
            qa_prompt_dict, trace, history_limit
        )
        # ^ render riwayat (sama seperti di HypothesisGen)

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]
        # ^ ambil semua Experiment dari riwayat: [(hyp, EXP, fb), ...]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)
        # ^ kumpulkan SEMUA FactorTask dari semua experiment sebelumnya
        #   ini dikirim ke LLM supaya dia tahu faktor apa yang sudah dicoba

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": function_lib_description,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": None,
        }, True     # True = json_mode
Method convert() (baris 416–591) — OVERRIDE convert dari parent — logika utama
Ini method terpanjang dan terpenting — tempat LLM dipanggil dan hasilnya divalidasi.


    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        history_limit = DEFAULT_HISTORY_LIMIT

        while history_limit >= MIN_HISTORY_LIMIT:
            try:
                return self._convert_with_history_limit(hypothesis, trace, history_limit)
            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                else:
                    raise

        return self._convert_with_history_limit(hypothesis, trace, MIN_HISTORY_LIMIT)
        # ^ flow yang sama dengan gen(): coba generate, kalau prompt terlalu panjang
        #   kurangi riwayat, coba lagi
Method _convert_with_history_limit() (baris 433–591) — INTI: generate + validasi faktor
Part 1 — render prompt:


    def _convert_with_history_limit(self, hypothesis, trace, history_limit):
        context, json_flag = self.prepare_context(hypothesis, trace, history_limit)

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.background,
                experiment_output_format=context["experiment_output_format"],
            )
        )
        # ^ system: "Given this scenario, generate factor expressions..."

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                function_lib_description=context["function_lib_description"],
                target_list=context["target_list"],
                RAG=context["RAG"],
                expression_duplication=None      # awalnya tidak ada feedback duplikasi
            )
        )
Part 2 — loop validasi (baris 460–584):


        flag = False                           # flag: semua faktor valid?
        expression_duplication_prompt = None    # feedback duplikasi (kumulatif)

        while True:
            if flag:
                break
                # ^ semua faktor sudah lolos validasi → keluar loop

            resp = self._call_llm(user_prompt, system_prompt, json_flag)
            # ^ PANGGIL LLM! generate faktor-faktor

            try:
                response_dict = robust_json_parse(resp)
                # ^ parse JSON response → dict
                #   misal: {"vol_mom": {"description": "...", "expression": "Std(Close,20)/Mean(Volume,10)", ...}}
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}, retrying...")
                continue
                # ^ JSON rusak → coba lagi (loop kembali ke while)

            proposed_names = []        # nama faktor yang lolos validasi
            proposed_exprs = []        # ekspresi faktor yang lolos validasi

            for i, factor_name in enumerate(response_dict):
                factor_data = response_dict.get(factor_name, {})
                if not isinstance(factor_data, dict):
                    continue
                expr = factor_data.get("expression", "")
                description = factor_data.get("description", "")
                formulation = factor_data.get("formulation", "")
                variables = factor_data.get("variables", {})
Part 2a — cek parsability:


                if not self.factor_regulator.is_parsable(expr):
                    logger.info(f"Failed to parse expr: {expr}, retrying...")
                    break
                    # ^ ekspresi tidak bisa di-parse oleh AST parser
                    #   break → keluar for loop → while loop mulai lagi
                    #   LLM dipanggil ulang untuk generate yang baru
Part 2b — evaluasi duplikasi dan kompleksitas:


                success, eval_dict = self.factor_regulator.evaluate(expr)
                # ^ evaluate() melakukan:
                #   1. parse AST dari ekspresi
                #   2. match_alphazoo: cek subtree mana yang sudah ada di alpha zoo
                #   3. hitung: num_free_args, num_unique_vars, num_all_nodes
                #   4. hitung: symbol_length, num_base_features
                #   return: (success, dict metrik)

                if not success:
                    break
                    # ^ evaluasi gagal → regenerate
Part 2c — consistency check (opsional):


                if self.consistency_enabled and self.quality_gate is not None:
                    try:
                        passed, feedback, results = self.quality_gate.evaluate(
                            hypothesis=str(hypothesis),
                            factor_name=factor_name,
                            factor_description=description,
                            factor_formulation=formulation,
                            factor_expression=expr,
                            variables=variables
                        )
                        # ^ cek apakah faktor KONSISTEN dengan hipotesis
                        #   misal: hipotesis tentang "momentum" tapi faktor pakai "volatility"
                        #   → inconsistent

                        if results.get("corrected_expression") and results["corrected_expression"] != expr:
                            expr = results["corrected_expression"]
                            factor_data["expression"] = expr
                            # ^ quality gate mungkin memperbaiki ekspresi
                            #   misal: salah fungsi → ganti yang benar

                            # re-check ekspresi yang sudah diperbaiki
                            if not self.factor_regulator.is_parsable(expr):
                                break
                            success, eval_dict = self.factor_regulator.evaluate(expr)
                            if not success:
                                break

                        if not passed:
                            logger.warning(f"Consistency check failed: {factor_name}")
                            # ^ warning tapi TIDAK break — faktor tetap diproses
                    except Exception as e:
                        logger.warning(f"Consistency check error: {e}")
Part 2d — cek acceptability (duplikasi + kompleksitas):


                if not self.factor_regulator.is_expression_acceptable(eval_dict):
                    # ^ faktor TIDAK acceptable: terlalu mirip alpha zoo, atau terlalu kompleks

                    num_all_nodes = eval_dict['num_all_nodes']
                    free_args_ratio = float(eval_dict['num_free_args']) / float(num_all_nodes) if num_all_nodes > 0 else 0.0
                    unique_vars_ratio = float(eval_dict['num_unique_vars']) / float(num_all_nodes) if num_all_nodes > 0 else 0.0
                    symbol_length = eval_dict.get('symbol_length', 0)
                    num_base_features = eval_dict.get('num_base_features', 0)
                    # ^ hitung rasio untuk feedback ke LLM

                    feedback_item = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["expression_duplication"])
                        .render(
                            prev_expression=expr,
                            duplicated_subtree_size=eval_dict['duplicated_subtree_size'],
                            duplication_threshold=self.factor_regulator.duplication_threshold,
                            duplicated_subtree=eval_dict.get('duplicated_subtree', ''),
                            matched_alpha=eval_dict.get('matched_alpha', ''),
                            free_args_ratio=free_args_ratio,
                            ...
                        )
                    )
                    # ^ render feedback template: beritahu LLM apa yang salah
                    #   misal: "Your expression Std(Close,20)/Mean(Volume,10) has
                    #   duplicated subtree 'Std(Close,20)' matching alpha_023
                    #   (size=5, threshold=8). Please generate a more original expression."

                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = '\n\n'.join(
                            [expression_duplication_prompt, feedback_item]
                        )
                    else:
                        expression_duplication_prompt = feedback_item
                    # ^ kumpulkan feedback: kalau beberapa faktor gagal,
                    #   semua feedback digabung

                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            ...
                            expression_duplication=expression_duplication_prompt
                            # ^ SISIPKAN feedback duplikasi ke user prompt!
                        )
                    )
                    break
                    # ^ break for loop → while loop mulai lagi
                    #   LLM dipanggil ulang DENGAN feedback duplikasi di prompt
                    #   ini membuat LLM "belajar" dari kesalahan dan generate yang lebih original
Part 2e — faktor lolos semua validasi:


                else:
                    proposed_names.append(factor_name)
                    proposed_exprs.append(expr)
                    # ^ faktor valid → tambahkan ke list yang lolos

                    if i == len(response_dict) - 1:
                        flag = True
                        # ^ ini faktor terakhir → semua lolos → set flag=True
                        #   while loop akan break di iterasi berikutnya
                    else:
                        continue
                        # ^ masih ada faktor lain → lanjut for loop
Part 3 — daftarkan faktor ke regulator (baris 587–591):


        self.factor_regulator.add_factor(proposed_names, proposed_exprs)
        # ^ tambahkan faktor yang lolos ke "alpha zoo" internal
        #   supaya faktor di round berikutnya tidak menduplikasi yang ini

        return self.convert_response(resp, trace)
        # ^ parse JSON terakhir → buat FactorExperiment
Visualisasi loop validasi:


LLM generate {"factor_A": {...}, "factor_B": {...}}
│
├─ factor_A: parsable? ✓ → evaluate? ✓ → acceptable? ✗ (duplikat!)
│   → feedback: "factor_A duplicates alpha_023"
│   → break, regenerate dengan feedback
│
LLM generate {"factor_C": {...}, "factor_D": {...}}  (BARU, dengan feedback)
│
├─ factor_C: parsable? ✓ → evaluate? ✓ → acceptable? ✓ → proposed!
├─ factor_D: parsable? ✓ → evaluate? ✓ → acceptable? ✓ → proposed!
│   → flag = True → break
│
└─ add_factor(["factor_C", "factor_D"], [...])
   └─ return convert_response(resp, trace)
Method convert_response() (baris 594–634) — parse JSON → FactorExperiment

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = robust_json_parse(response)
        tasks = []

        for factor_name in response_dict:
            factor_data = response_dict.get(factor_name, {})
            if not isinstance(factor_data, dict):
                continue
            description = factor_data.get("description", "")
            formulation = factor_data.get("formulation", "")
            expression = factor_data.get("expression", "")
            variables = factor_data.get("variables", {})
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    factor_expression=expression,
                    variables=variables,
                )
            )
        # ^ setiap entry di JSON → satu FactorTask
Sekarang FactorTask dipakai — sekilas:

Sekilas factors/coder/factor.py — FactorTask:


class FactorTask(CoSTEERTask):
    def __init__(self, factor_name, factor_description, factor_formulation,
                 factor_expression=None, variables={}, resource=None,
                 factor_implementation=False):
        self.factor_name = factor_name             # "vol_mom_ratio"
        self.factor_description = factor_description   # "Volatility-momentum ratio..."
        self.factor_formulation = factor_formulation   # "sigma_20 / mu_volume_10"
        self.factor_expression = factor_expression     # "Std(Close,20)/Mean(Volume,10)"
        self.variables = variables                     # {"Close": "closing price", ...}
        self.factor_resources = resource
        self.factor_implementation = factor_implementation  # False = belum diimplementasi
        super().__init__(name=factor_name)

    def get_task_information(self):
        return f"factor_name: {self.factor_name}\n..."
        # ^ format info untuk disisipkan ke prompt coder
Kembali ke convert_response():


        exp = QlibFactorExperiment(tasks)
        # ^ buat Experiment baru dengan list FactorTask

        exp.based_experiments = [QlibFactorExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]
        # ^ based_experiments: experiment sebelumnya yang sukses (feedback = True)
        #   [kosong] + [exp dari riwayat yang decision=True]
        #   dipakai untuk cek duplikasi nama faktor

        unique_tasks = []
        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)
        # ^ filter: buang FactorTask yang namanya sudah ada di experiment sebelumnya
        #   ini mencegah LLM menghasilkan faktor dengan nama yang sama

        exp.tasks = unique_tasks
        return exp
BAGIAN 9: class BacktestHypothesis2FactorExpression (baris 638–687)
Dipakai oleh FactorBackTestBasePropSetting — load faktor dari CSV file.


class BacktestHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, factor_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_path = factor_path
        # ^ path ke CSV file berisi faktor yang sudah jadi

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        if os.path.exists(self.factor_path):
            tasks = []
            factor_df = pd.read_csv(self.factor_path, usecols=["factor_name", "factor_expression"])
            # ^ baca CSV: hanya ambil kolom nama dan ekspresi

            for index, row in factor_df.iterrows():
                tasks.append(
                    FactorTask(
                        factor_name=row["factor_name"],
                        factor_description="",
                        factor_formulation="",
                        factor_expression=row["factor_expression"],
                        variables="",
                    )
                )
            # ^ setiap baris CSV → satu FactorTask
            #   description dan formulation kosong (tidak penting untuk backtest)

            exp = QlibFactorExperiment(tasks)
            # ... deduplikasi yang sama seperti di atas ...
            return exp
        else:
            raise ValueError(f"File {self.factor_csv_path} does not exist.")
            # ^ TIDAK panggil LLM sama sekali
            #   hanya load faktor dari file → langsung return
Ringkasan file ini
Class	Dipanggil kapan	Apa yang dilakukan
AlphaAgentHypothesisGen	AlphaAgentLoop.factor_propose()	LLM generate hipotesis dari direction + riwayat
AlphaAgentHypothesis2FactorExpression	AlphaAgentLoop.factor_construct()	LLM generate ekspresi faktor + validasi duplikasi/kompleksitas (loop sampai valid)
QlibFactorHypothesisGen	Mode tradisional (bukan AlphaAgent)	Versi basic tanpa direction
QlibFactorHypothesis2Experiment	Mode tradisional	Versi basic tanpa regulator
EmptyHypothesisGen	BacktestLoop	Return hipotesis kosong
BacktestHypothesis2FactorExpression	BacktestLoop	Load faktor dari CSV, tanpa LLM
File yang sudah ditelusuri dari sini:

File	Yang dijelaskan
components/proposal/init.py	LLMHypothesisGen, FactorHypothesisGen, LLMHypothesis2Experiment, FactorHypothesis2Experiment
core/prompts.py	Prompts (Singleton + dict dari YAML)
factors/coder/factor.py	FactorTask, FactorExperiment (alias)
factors/regulator/factor_regulator.py	FactorRegulator (parse, evaluate, cek duplikasi)
factors/experiment.py	QlibFactorExperiment, QlibAlphaAgentScenario
Belum ditelusuri:

factors/feedback.py → AlphaAgentQlibFactorHypothesisExperiment2Feedback
factors/qlib_coder.py → QlibFactorParser
factors/runner.py → QlibFactorRunner
pipeline/planning.py → generate_parallel_directions, load_run_config
pipeline/evolution/mutation.py → MutationOperator
pipeline/evolution/crossover.py → CrossoverOperator
factors/latent_proposal.py → Latent variants
llm/client.py → LocalLLMBackend, robust_json_parse
factors/library.py → FactorLibraryManager