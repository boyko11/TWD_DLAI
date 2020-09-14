import unittest
from model_service import ModelService
import numpy as np
from td_utils import *
import IPython


class ModelServiceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_service = ModelService()
        cls.activates, cls.negatives, cls.backgrounds = load_raw_audio()
        pass

    def test_is_overlapping(self):

        overlap1 = self.model_service.is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
        self.assertFalse(overlap1)

        overlap2 = self.model_service.is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
        self.assertTrue(overlap2)

    def test_insert_audio_clip(self):

        np.random.seed(5)
        audio_clip, segment_time = self.model_service.insert_audio_clip(self.backgrounds[0], self.activates[0], [(3790, 4400)])
        audio_clip.export("insert_test.wav", format="wav")
        print("Segment Time: ", segment_time)
        IPython.display.Audio("insert_test.wav")

        expected_segment_time = (2915, 3635)

        self.assertEqual(expected_segment_time, segment_time)

    def test_insert_ones(self):

        arr1 = self.model_service.insert_ones(np.zeros((1, self.model_service.Ty)), 9700)
        plt.plot(self.model_service.insert_ones(arr1, 4251)[0, :])
        plt.show()
        print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])

        self.assertAlmostEqual(0.0, arr1[0][1333])
        self.assertAlmostEqual(1.0, arr1[0][634])

    def test_create_training_example(self):

        x, y = self.model_service.create_training_example(self.backgrounds[0], self.activates, self.negatives)



if __name__ == '__main__':
    unittest.main()
