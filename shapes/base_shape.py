from abc import ABC, abstractmethod

class ShapeBase(ABC):
    @abstractmethod
    def add_mesh(self, builder):
        pass
    
    @abstractmethod
    def add_spheres(self, builder):
        pass
