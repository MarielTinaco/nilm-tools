from __future__ import annotations

import json
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Any
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

import src.utils.paths_manager as pathsman

class ProfileHandlerTypes(Enum):
        UNETNILM = "unetnilm"

@dataclass
class DatasetProfile:
        parent_dir : Union[Path, str]
        handler : str
        dataset_name : str
        metadata : dict
        mode : str = "w"
        full_path : Path = field(default_factory=Path)

        def __post_init__(self):

                if self.mode == "w":
                        self.full_path = Path(self.parent_dir) / \
                                        f'{self.handler}_{self.dataset_name}_{datetime.now():20%y%m%d_%H%M%S}'

                        self.full_path.mkdir(exist_ok=True)

                        with open(self.full_path / "metadata.json", "w", encoding="utf8") as jfile:
                                json.dump(self.metadata, jfile, indent=4)

        @classmethod
        def extract(cls, profile_path : Path) -> 'DatasetProfile':
                
                if not profile_path.exists():
                        raise "Profile doesn't exist"

                handler = profile_path.name.split("_")[0]
                dataset_name = profile_path.name.split("_")[1]

                with open(profile_path / "metadata.json", "r") as jfile:
                        metadata = json.load(jfile)

                mode = "r"
                parent_dir = profile_path.parent

                return cls(parent_dir, handler, dataset_name, metadata, mode, full_path=profile_path)


class ProfileHandlerContext(object):

        def __init__(self, strategy: "ProfileHandlingStrategy"):
                
                self.strategy = strategy
                self.profiles_dir = strategy.profiles_dir

        def write(self, *data, **kwargs):
                if not kwargs.get("metadata"):
                        kwargs["metedata"] = {}

                if not kwargs.get("dataset_name"):
                        kwargs["dataset_name"] = "unspecified_source"

                kwargs["profiles_dir"] = self.profiles_dir

                return self.strategy.write(*data, **kwargs)

        def read(self, path, *args, **kwargs):
                path = Path(path)

                if not self.profiles_dir.exists():
                        raise NotADirectoryError(f"Provided profiles directory: {str(self.profiles_dir)} does not exist")

                if not path.exists():
                        raise FileNotFoundError(f"No profile found in {str(path)}")

                return self.strategy.read(path, *args, **kwargs)


class ProfileHandlingStrategy(ABC):

        def __init__(self, profiles_dir, *args, **kwargs):
                self._profiles_dir = profiles_dir

        @property
        def profiles_dir(self):
                return self._profiles_dir

        @abstractmethod
        def write(self, *data : Any, **kwargs) -> DatasetProfile:
                raise NotImplementedError

        @abstractmethod
        def read(self, path : Union[Path, str], *args, **kwargs) -> Any:
                raise NotImplementedError


class UNETNiLMProfileHandler(ProfileHandlingStrategy):

        def __init__(self, profiles_dir, *args, **kwargs):
                super(UNETNiLMProfileHandler, self).__init__(profiles_dir, *args, **kwargs)
                self.subdir = kwargs["subdir"]

        def write(self, *data : Any, **kwargs):
                assert len(data) == 4, "Please provide [denoise_inputs, noise_inputs, targets, states]"
                denoise_inputs = data[0]
                noise_inputs = data[1]
                targets = data[2]
                states = data[3]

                profile = DatasetProfile(parent_dir= self._profiles_dir,
                                         handler= kwargs.get("handler"),
                                         dataset_name= kwargs.get("dataset_name"),
                                         metadata= kwargs.get("metadata"),
                                         mode= "w")

                subdir = profile.full_path / self.subdir
                subdir.mkdir(exist_ok=True)

                np.save(str(subdir) + "/denoise_inputs.npy", denoise_inputs)
                np.save(str(subdir) + "/noise_inputs.npy", noise_inputs)
                np.save(str(subdir) + "/targets.npy", targets)
                np.save(str(subdir) + "/states.npy", states)

                return profile                


        def read(self, profile_path : Union[Path, str] = None, *args, **kwargs) -> Any:
                 
                subdir = Path(profile_path) / self.subdir
                if not subdir.exists():
                        raise NotADirectoryError()
                
                x1 = np.load(str(subdir / "denoise_inputs.npy"))
                x2 = np.load(str(subdir / "noise_inputs.npy"))
                y  = np.load(str(subdir / "targets.npy"))
                z  = np.load(str(subdir / "states.npy"))

                return DatasetProfile.extract(profile_path), x1, x2, y, z


ProfileHandlingRegistry = {
        ProfileHandlerTypes.UNETNILM : UNETNiLMProfileHandler
}

class ProfileHandler(object):

        @staticmethod
        def mkdir(profile_path, *args, **kwargs):
                Path(profile_path).mkdir(*args, **kwargs)

        @staticmethod
        def write(*data, profile : DatasetProfile, **kwargs):
                
                if profile.handler not in [i.value for i in ProfileHandlerTypes]:
                        raise f"{profile.handler} Handling Not Supported"

                Handler = ProfileHandlingRegistry[ProfileHandlerTypes(profile.handler)]
                ctx = ProfileHandlerContext(strategy=Handler(profiles_dir=profile.parent_dir, **kwargs))

                return ctx.write(*data,
                                 handler=profile.handler,
                                 dataset_name=profile.dataset_name,
                                 metadata=profile.metadata)
        
        @staticmethod
        def read(profile_path, *args, **kwargs):

                profile = DatasetProfile.extract(Path(profile_path))

                if profile.handler not in [i.value for i in ProfileHandlerTypes]:
                        raise f"{profile.handler} Handling Not Supported"
                
                Handler = ProfileHandlingRegistry[ProfileHandlerTypes(profile.handler)]
                ctx = ProfileHandlerContext(strategy=Handler(profiles_dir=profile.parent_dir, **kwargs))

                return ctx.read(profile_path, *args, **kwargs)