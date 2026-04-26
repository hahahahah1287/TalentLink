# -*- coding: utf-8 -*-
"""
legal_parser 单元测试

验证结构化切分器能正确解析法律文档：
- 条款数量
- metadata 完整性（law、chapter、article、source）
- 章节归属正确性
"""
import os
import sys
import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.legal_parser import LegalDocumentParser, parse_legal_document


# ==================== Fixtures ====================

@pytest.fixture
def parser():
    return LegalDocumentParser()


@pytest.fixture
def labor_law_text():
    """加载真实的 labor_law.txt"""
    law_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "labor_law.txt"
    )
    with open(law_path, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def simple_law_text():
    """最小化测试文本"""
    return """第一章　总则

　　第一条　为了保护劳动者的合法权益，制定本法。

　　第二条　在中华人民共和国境内的企业适用本法。

第二章　劳动合同

　　第三条　劳动合同是确立劳动关系的协议。

　　第四条　订立劳动合同应当遵循平等自愿原则。
"""


# ==================== 基础解析测试 ====================

class TestBasicParsing:
    """基础解析功能测试"""

    def test_parse_simple_text(self, parser, simple_law_text):
        """简单文本能正确切分为 4 个条款"""
        docs = parser.parse(simple_law_text, source="test.txt")
        assert len(docs) == 4, f"期望 4 个条款，实际 {len(docs)}"

    def test_article_content_preserved(self, parser, simple_law_text):
        """条款内容完整保留"""
        docs = parser.parse(simple_law_text, source="test.txt")
        assert "保护劳动者的合法权益" in docs[0].page_content
        assert "平等自愿原则" in docs[3].page_content

    def test_empty_text(self, parser):
        """空文本返回空列表"""
        docs = parser.parse("", source="empty.txt")
        assert len(docs) == 0

    def test_no_chapter_text(self, parser):
        """无章节结构的文本"""
        text = "　　第一条　这是一条法律。\n\n　　第二条　这是另一条法律。"
        docs = parser.parse(text, source="simple.txt")
        assert len(docs) == 2


# ==================== Metadata 测试 ====================

class TestMetadata:
    """元数据正确性测试"""

    def test_metadata_fields_present(self, parser, simple_law_text):
        """每个文档都包含必需的 metadata 字段"""
        docs = parser.parse(simple_law_text, source="test.txt")
        required_fields = {"law", "chapter", "article", "source"}
        for doc in docs:
            assert required_fields.issubset(doc.metadata.keys()), \
                f"缺少 metadata 字段: {required_fields - doc.metadata.keys()}"

    def test_source_preserved(self, parser, simple_law_text):
        """source 字段正确传递"""
        docs = parser.parse(simple_law_text, source="labor_law.txt")
        for doc in docs:
            assert doc.metadata["source"] == "labor_law.txt"

    def test_article_number_extracted(self, parser, simple_law_text):
        """条款号正确提取"""
        docs = parser.parse(simple_law_text, source="test.txt")
        article_numbers = [doc.metadata["article"] for doc in docs]
        assert "第一条" in article_numbers
        assert "第二条" in article_numbers
        assert "第三条" in article_numbers
        assert "第四条" in article_numbers

    def test_chapter_assignment(self, parser, simple_law_text):
        """章节归属正确"""
        docs = parser.parse(simple_law_text, source="test.txt")
        # 第一条、第二条属于第一章
        assert "第一章" in docs[0].metadata["chapter"]
        assert "第一章" in docs[1].metadata["chapter"]
        # 第三条、第四条属于第二章
        assert "第二章" in docs[2].metadata["chapter"]
        assert "第二章" in docs[3].metadata["chapter"]


# ==================== 真实数据测试 ====================

class TestRealLaborLaw:
    """使用真实 labor_law.txt 的集成测试"""

    def test_article_count(self, labor_law_text):
        """劳动法应解析为约 107 个条款"""
        docs = parse_legal_document(labor_law_text, source="labor_law.txt")
        # 劳动法共 107 条
        assert 100 <= len(docs) <= 115, \
            f"期望约 107 个条款，实际 {len(docs)}"

    def test_first_article(self, labor_law_text):
        """第一条内容和元数据正确"""
        docs = parse_legal_document(labor_law_text, source="labor_law.txt")
        first = docs[0]
        assert first.metadata["article"] == "第一条"
        assert "第一章" in first.metadata["chapter"]
        assert "保护劳动者的合法权益" in first.page_content

    def test_last_article(self, labor_law_text):
        """最后一条（第一百零七条）正确解析"""
        docs = parse_legal_document(labor_law_text, source="labor_law.txt")
        last = docs[-1]
        assert "一百零七" in last.metadata["article"]
        assert "１９９５年" in last.page_content or "施行" in last.page_content

    def test_chapter_distribution(self, labor_law_text):
        """各章节均有条款分布"""
        docs = parse_legal_document(labor_law_text, source="labor_law.txt")
        chapters = set(doc.metadata["chapter"] for doc in docs)
        # 劳动法共 13 章
        assert len(chapters) >= 10, \
            f"期望覆盖 10+ 章节，实际 {len(chapters)} 章: {chapters}"

    def test_chapter_44_belongs_to_chapter_4(self, labor_law_text):
        """第四十四条应属于第四章（工作时间和休息休假）"""
        docs = parse_legal_document(labor_law_text, source="labor_law.txt")
        art44 = [d for d in docs if d.metadata["article"] == "第四十四条"]
        assert len(art44) == 1
        assert "第四章" in art44[0].metadata["chapter"]

    def test_no_empty_content(self, labor_law_text):
        """所有文档内容不为空"""
        docs = parse_legal_document(labor_law_text, source="labor_law.txt")
        for doc in docs:
            assert doc.page_content.strip(), \
                f"{doc.metadata['article']} 内容为空"


# ==================== 法律名称提取测试 ====================

class TestLawNameExtraction:
    """法律名称提取测试"""

    def test_extract_from_book_marks(self, parser):
        """从《书名号》中提取法律名称"""
        text = "根据《中华人民共和国劳动合同法》制定本条例。\n　　第一条　测试。"
        docs = parser.parse(text, source="test.txt")
        assert docs[0].metadata["law"] == "中华人民共和国劳动合同法"

    def test_fallback_to_filename(self, parser):
        """无书名号时回退到文件名"""
        text = "　　第一条　为了保护劳动者权益，制定本法。"
        docs = parser.parse(text, source="labor_law.txt")
        assert docs[0].metadata["law"] == "labor_law"


# ==================== 便捷函数测试 ====================

def test_convenience_function(simple_law_text):
    """parse_legal_document 便捷函数可正常使用"""
    docs = parse_legal_document(simple_law_text, source="test.txt")
    assert len(docs) == 4
    assert all("article" in d.metadata for d in docs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
