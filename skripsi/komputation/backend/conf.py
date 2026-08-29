from __future__ import annotations

# [terjawab — rangkuman file ini]:
#   conf.py menyediakan BASIS konfigurasi berbasis pydantic-settings. Inti:
#   1. ExtendedBaseSettings  : kelas dasar; tiap field bisa diisi dari ENVIRONMENT VARIABLE.
#   2. env_prefix            : awalan nama env var per kelas (mis. "QLIB_FACTOR_").
#   3. ExtendedEnvSettingsSource : memperluas pencarian env var agar juga melihat prefix
#                              kelas INDUK, bukan hanya kelas sendiri (lihat get_field_value).
#   4. RDAgentSettings       : setting global path workspace, cache pickle, multiprocessing.
#   Alur baca nilai field:  init kwargs  →  ENV (prefix kelas → prefix induk)  →  default.
#   Contoh: field `latent_steps` di kelas ber-prefix "QLIB_FACTOR_" dibaca dari
#           env var QLIB_FACTOR_LATENT_STEPS; bila tak ada, pakai default di kode.
# TODO: use pydantic for other modules in Qlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class ExtendedEnvSettingsSource(EnvSettingsSource):
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Dynamically gather prefixes from the current and parent classes
        # [terjawab]: env_prefix = awalan nama environment variable untuk kelas setting.
        #   mis. env_prefix="QLIB_FACTOR_" → field `latent_steps` dibaca dari env
        #   QLIB_FACTOR_LATENT_STEPS. Di sini prefix kelas sendiri + prefix kelas induk dikumpulkan.
        prefixes = [self.config.get("env_prefix", "")] #* prefix dari class itu sendiri(inisiasi di settings.py)
        if hasattr(self.settings_cls, "__bases__"): #* cek parent class
            for base in self.settings_cls.__bases__:
                if hasattr(base, "model_config"):
                    parent_prefix = base.model_config.get("env_prefix")
                    if parent_prefix and parent_prefix not in prefixes:
                        prefixes.append(parent_prefix) #* tambahkan prefix parent class jika ada dan belum ada di list
        for prefix in prefixes:
            self.env_prefix = prefix
            env_val, field_key, value_is_complex = super().get_field_value(field, field_name) #* cari di env dengan prefix yang sudah di-set, jika tidak ditemukan lanjut ke prefix berikutnya
            if env_val is not None:
                return env_val, field_key, value_is_complex #* return yang pertama ditemukan

        return super().get_field_value(field, field_name)


class ExtendedSettingsConfigDict(SettingsConfigDict, total=False): ...


class ExtendedBaseSettings(BaseSettings):

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,  # noqa
        env_settings: PydanticBaseSettingsSource,  # noqa
        dotenv_settings: PydanticBaseSettingsSource,  # noqa
        file_secret_settings: PydanticBaseSettingsSource,  # noqa
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (ExtendedEnvSettingsSource(settings_cls),)
        #* ^ semua field akan dicari di env dengan prefix yang sesuai, mulai dari class itu sendiri sampai parent class


class RDAgentSettings(ExtendedBaseSettings):
    # TODO: (xiao) I think LLMSetting may be a better name.
    # TODO: (xiao) I think most of the config should be in oai.config
    # Log configs
    # TODO: (xiao) think it can be a separate config.
    log_trace_path: str | None = None

    # azure document intelligence configs
    azure_document_intelligence_key: str = ""
    azure_document_intelligence_endpoint: str = ""
    # factor extraction conf
    max_input_duplicate_factor_group: int = 300
    max_output_duplicate_factor_group: int = 20
    max_kmeans_group_number: int = 40

    # Default absolut berdasarkan lokasi file ini — tidak bergantung CWD maupun DATA_RESULTS_DIR
    # Override via env var WORKSPACE_PATH atau PICKLE_CACHE_FOLDER_PATH_STR
    _abs_data_root: ClassVar[Path] = Path(__file__).resolve().parent.parent / "data" / "results"

    workspace_path: Path = _abs_data_root / "workspace"

    # multi processing conf
    multi_proc_n: int = 1

    # pickle cache conf
    cache_with_pickle: bool = True
    pickle_cache_folder_path_str: str = str(_abs_data_root / "pickle_cache")

    use_file_lock: bool = (
        True  # when calling the function with same parameters, whether to use file lock to avoid
        # executing the function multiple times
    )

# [terjawab — lihat rangkuman + diagram alur di header file ini (atas)].
RD_AGENT_SETTINGS = RDAgentSettings()
