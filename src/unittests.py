import unittest
from flask import Flask
from flask_testing import TestCase
from io import BytesIO
from api import app




class UnitTests(TestCase):
    """Unit tests for the API endpoints."""

    def create_app(self):
        """Create and configure a Flask app for testing."""
        app.config['TESTING'] = True
        return app

    def test_create_dataset(self):
        """Test '/create_dataset' endpoint. Expects a 200 response."""
        response = self.client.get('/create_dataset')
        self.assertEqual(response.status_code, 200)

    def test_training(self):
        """Test '/training' endpoint. Expects a 200 response."""
        response = self.client.post('/training')
        self.assertEqual(response.status_code, 200)

    def test_testing(self):
        """Test '/testing' endpoint. Expects a 200 response."""
        response = self.client.post('/testing')
        self.assertEqual(response.status_code, 200)




if __name__ == '__main__':
    # Append tests 
    suite = unittest.TestSuite()
    suite.addTest(UnitTests('test_create_dataset'))
    suite.addTest(UnitTests('test_training'))
    suite.addTest(UnitTests('test_testing'))

    # Run the test suite
    unittest.TextTestRunner().run(suite)
