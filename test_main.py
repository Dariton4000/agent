#!/usr/bin/env python3
"""
Test suite for the AI Research Assistant.

This module contains unit tests and integration tests for the main functionality.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import requests
from main import AIResearchAssistant


class TestAIResearchAssistant(unittest.TestCase):
    """Test cases for AIResearchAssistant class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.assistant = AIResearchAssistant()
    
    def test_print_fragment(self):
        """Test the print_fragment method."""
        # Create a mock fragment
        fragment = Mock()
        fragment.content = "Test content"
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            self.assistant.print_fragment(fragment)
            mock_print.assert_called_once_with("Test content", end="", flush=True)
    
    @patch('main.AsyncWebCrawler')
    async def test_crawl_webpage_success(self, mock_crawler_class):
        """Test successful webpage crawling."""
        # Setup mock
        mock_crawler = AsyncMock()
        mock_result = Mock()
        mock_result.markdown = "# Test Markdown Content"
        mock_crawler.arun.return_value = mock_result
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        
        # Test
        result = await self.assistant.crawl_webpage("https://example.com")
        
        # Assertions
        self.assertEqual(result, "# Test Markdown Content")
        mock_crawler.arun.assert_called_once()
    
    @patch('main.AsyncWebCrawler')
    async def test_crawl_webpage_failure(self, mock_crawler_class):
        """Test webpage crawling failure."""
        # Setup mock to raise exception
        mock_crawler_class.side_effect = Exception("Network error")
        
        # Test
        with self.assertRaises(Exception):
            await self.assistant.crawl_webpage("https://example.com")
    
    def test_crawl_sync_success(self):
        """Test synchronous crawl wrapper success."""
        with patch.object(self.assistant, 'crawl_webpage') as mock_crawl:
            mock_crawl.return_value = "Test content"
            
            with patch('asyncio.run') as mock_run:
                mock_run.return_value = "Test content"
                result = self.assistant.crawl_sync("https://example.com")
                
                self.assertEqual(result, "Test content")
                mock_run.assert_called_once()
    
    def test_crawl_sync_failure(self):
        """Test synchronous crawl wrapper failure."""
        with patch('asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Network error")
            result = self.assistant.crawl_sync("https://example.com")
            
            self.assertIn("Error crawling webpage", result)
    
    @patch('requests.get')
    def test_wikipedia_search_success(self, mock_get):
        """Test successful Wikipedia search."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'query': {
                'search': [
                    {'title': 'Python (programming language)'},
                    {'title': 'Python (mythology)'},
                    {'title': 'Python (film)'}
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test
        result = self.assistant.wikipedia_search("Python", 3)
        
        # Assertions
        expected = ['Python (programming language)', 'Python (mythology)', 'Python (film)']
        self.assertEqual(result, expected)
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_wikipedia_search_failure(self, mock_get):
        """Test Wikipedia search failure."""
        mock_get.side_effect = requests.RequestException("API error")
        
        result = self.assistant.wikipedia_search("Python")
        
        self.assertEqual(result, [])
    
    @patch('requests.get')
    def test_wikipedia_search_limit_cap(self, mock_get):
        """Test that Wikipedia search limits are properly capped."""
        mock_response = Mock()
        mock_response.json.return_value = {'query': {'search': []}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test with limit over 500
        self.assistant.wikipedia_search("test", 1000)
        
        # Check that the API was called with limit 500
        call_args = mock_get.call_args[1]['params']
        self.assertEqual(call_args['srlimit'], 500)
    
    @patch('requests.get')
    def test_get_wikipedia_page_success(self, mock_get):
        """Test successful Wikipedia page retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'query': {
                'pages': {
                    '123': {
                        'extract': 'This is the page content.'
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.assistant.get_wikipedia_page("Python")
        
        self.assertEqual(result, "This is the page content.")
    
    @patch('requests.get')
    def test_get_wikipedia_page_not_found(self, mock_get):
        """Test Wikipedia page not found."""
        mock_response = Mock()
        mock_response.json.return_value = {'query': {'pages': {}}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.assistant.get_wikipedia_page("NonexistentPage")
        
        self.assertEqual(result, "No page found.")
    
    @patch('requests.get')
    def test_get_wikipedia_page_failure(self, mock_get):
        """Test Wikipedia page retrieval failure."""
        mock_get.side_effect = requests.RequestException("API error")
        
        result = self.assistant.get_wikipedia_page("Python")
        
        self.assertIn("Error fetching page", result)
    
    def test_setup_tools(self):
        """Test that tools are properly configured."""
        tools = self.assistant.setup_tools()
        
        self.assertEqual(len(tools), 3)
        
        # Check tool names
        tool_names = [tool.name for tool in tools]
        expected_names = [
            "Wikipedia Search matching pages",
            "Wikipedia fetches a page", 
            "Crawl a webpage"
        ]
        for name in expected_names:
            self.assertIn(name, tool_names)
    
    def test_calculate_context_usage(self):
        """Test context usage calculation."""
        # Create mocks
        mock_model = Mock()
        mock_chat = Mock()
        
        mock_model.apply_prompt_template.return_value = "formatted prompt"
        mock_model.tokenize.return_value = ["token1", "token2", "token3"]  # 3 tokens
        mock_model.get_context_length.return_value = 1000
        
        result = self.assistant.calculate_context_usage(mock_model, mock_chat)
        
        # 1000 / 3 = 333.33...
        self.assertAlmostEqual(result, 333.33, places=1)
    
    def test_calculate_context_usage_zero_tokens(self):
        """Test context usage calculation with zero tokens."""
        mock_model = Mock()
        mock_chat = Mock()
        
        mock_model.apply_prompt_template.return_value = ""
        mock_model.tokenize.return_value = []  # 0 tokens
        mock_model.get_context_length.return_value = 1000
        
        result = self.assistant.calculate_context_usage(mock_model, mock_chat)
        
        self.assertEqual(result, float('inf'))
    
    def test_calculate_context_usage_error(self):
        """Test context usage calculation error handling."""
        mock_model = Mock()
        mock_chat = Mock()
        
        mock_model.apply_prompt_template.side_effect = Exception("Error")
        
        result = self.assistant.calculate_context_usage(mock_model, mock_chat)
        
        self.assertEqual(result, 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    @patch('main.lms')
    @patch('builtins.input')
    def test_run_with_empty_input(self, mock_input, mock_lms):
        """Test running the assistant with empty input (should exit)."""
        # Setup mocks
        mock_input.return_value = ""  # Empty input to exit
        mock_lms.list_loaded_models.return_value = []
        mock_lms.list_downloaded_models.return_value = [Mock(model_key="test-model")]
        
        assistant = AIResearchAssistant()
        
        with patch.object(assistant, 'load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            with patch('main.lms.Chat') as mock_chat_class:
                mock_chat = Mock()
                mock_chat_class.return_value = mock_chat
                
                # This should not raise an exception
                assistant.run()
                
                mock_load.assert_called_once()


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def setUp(self):
        self.assistant = AIResearchAssistant()
    
    @patch('requests.get')
    def test_wikipedia_search_empty_response(self, mock_get):
        """Test Wikipedia search with empty response."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.assistant.wikipedia_search("test")
        
        self.assertEqual(result, [])
    
    @patch('requests.get')
    def test_get_wikipedia_page_malformed_response(self, mock_get):
        """Test Wikipedia page retrieval with malformed response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'query': {
                'pages': {
                    '123': {}  # No extract field
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.assistant.get_wikipedia_page("test")
        
        self.assertEqual(result, "No content found for the given page.")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
