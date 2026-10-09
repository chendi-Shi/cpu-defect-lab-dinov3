import base64
import io
import unittest
from PIL import Image
from lab import validation_partition
from demo_server import decode_image


class ValidationTests(unittest.TestCase):
    def test_partitions_disjoint_and_cover(self):
        a, b = validation_partition(42)
        self.assertEqual(len(a), 21)
        self.assertEqual(len(b), 21)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(sorted(a+b), list(range(42)))
        self.assertEqual((a,b), validation_partition(42))

    def test_image_decode(self):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(buffer, 'PNG')
        image = decode_image(base64.b64encode(buffer.getvalue()).decode())
        self.assertEqual(image.size, (8, 8))
        self.assertEqual(image.getpixel((0,0)), (255,0,0))

    def test_invalid_image(self):
        for payload in ['!!!!', base64.b64encode(b'not an image').decode(), None]:
            with self.assertRaises(ValueError):
                decode_image(payload)


if __name__ == '__main__':
    unittest.main()
