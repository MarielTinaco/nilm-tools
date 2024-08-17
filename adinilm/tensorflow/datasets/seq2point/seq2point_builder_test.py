

from .seq2point_builder import Seq2PointBuilder


if __name__ == "__main__":

        import adinilm.utils.paths_manager as pman


        ds = Seq2PointBuilder(pman.PROFILES_DIR / "unetnilm_ukdale_20240730_143151",
                              split= 0.7)
        
