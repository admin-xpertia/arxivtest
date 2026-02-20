from langchain_community.tools.arxiv.tool import ArxivQueryRun


def get_arxiv_tool() -> ArxivQueryRun:
    return ArxivQueryRun()
