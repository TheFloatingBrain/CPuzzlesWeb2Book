# Copywrite (c) Christpoher A. Greeley 2025, Licensed under
# Non - AI No Kill Do No Harm Licencse see accompanying LICENSE.md or NOTICE
# file and redistribute with LICENSE.md and NOTICE file
from typing import List, Tuple, Dict
from dataclasses import dataclass
import math
import copy
from pathlib import Path
import tqdm
import re
import time
import random
import requests
import sys
import os
import argparse
import glob
import threading
import concurrent.futures
from bs4 import BeautifulSoup, Tag
import pdfkit
from pypdf import PdfReader, PdfWriter
import pypdf


ROOT_URL = "https://programmedlessons.org/CPuzzles/" # Content retrieved via this url credit Bradley Kjell
PARSER = "html.parser"
C_PUZZLERS = "C Puzzlers"
LOCAL_ROOT = Path(".") / "local" / "out"
LOCAL_SOURCE = Path(".") / "local" / "source"

HTML_TEMPLATE_FORMAT_STRING = """
<html>
<link href="../../PuzzleStyle.css" rel="stylesheet" type="text/css"/>
<title>{}</title><head></head><body>{}</body>
</html>
"""

C_KEYWORDS = [
    "const ",
    "return ",
    "int ",
    "void ",
    "float ",
    "double ",
    "bool ",
    "char ",
    "unsigned ",
    "signed ",
    "long ",
    "short ",
    "true",
    "false",
    "typedef",
    "struct ",
    "union ",
    "enum ",
    "for ",
    "while ",
    "static ",
    "extern ",
    "volatile ",
    "inline ",
    "break",
    "if ",
    "else",
    "goto "
    "#include",
    "#define",
    "#pragma"
]


KEYWORD_FORMAT_STRING = "<strong>{}</strong>"


def str_find_all(string: str, substring: str, limit: int = 1000) -> List[int]:
    result = string.find(substring)
    results = []
    max_count = 0
    while result > 0 and max_count < limit:
        results.append(result)
        result = string.find(substring[result:])
        max_count += 1
    if max_count >= limit:
        print(string)
        print(results)
    return results


def str_extract_betweens(text: str, begin: str, end: str) -> List[str]:
    begins = str_find_all(text, begin)
    ends = str_find_all(text, end)
    if len(begins) != len(ends):
        minimum_length = min(len(begins), len(ends))
        begins = begins[:minimum_length]
        ends = ends[:minimum_length]
    return [
        text[a + len(begin):b]
        for a, b in zip(begins, ends)
    ]


def bold_syntax(code: str, keywords: List[str]) -> str:
    comments = str_extract_betweens(code, "/*", "*/")
    for ii, comment in enumerate(comments):
        code.replace(comment, f"/*$${ii}*/")
    for keyword in keywords:
        code = code.replace(keyword, KEYWORD_FORMAT_STRING.format(keyword))
    for ii, comment in enumerate(comments):
        code.replace(f"/*$${ii}*/", comment)
    return code


def bold_codes_syntax(code: str, keywords: List[str] = C_KEYWORDS) -> str:
    code_segments = str_extract_betweens(code, "<pre>", "</pre>")
    for code_segment in code_segments:
        original = code_segment
        new_code = bold_syntax(original, keywords)
        code = code.replace(original, new_code)
    return code


def httpsify(path_url: str | Path) -> str:
    return str(path_url).replace("https:/", "https://")


def extract_body(page) -> Tuple[BeautifulSoup, str]:
    if page.html:
        page.html.unwrap()
    if page.title:
        title = page.title.contents[0]
        page.title.unwrap()
    else:
        title = None
    if page.head:
        page.head.unwrap()
    if page.body:
        page.body.unwrap()
    return page, title


def q_top_page(soup_q_body, letter_part, seg_num) -> List:
    body_str = str(soup_q_body)
    header_index = body_str.find("<h4>")
    trimmed = body_str[:header_index - 1]
    first_q_tag_begin = trimmed.find("<a")
    if first_q_tag_begin > 0:
        first_q_tag_end = trimmed[first_q_tag_begin:].find(">")
        tag = trimmed[first_q_tag_begin:
                      (first_q_tag_begin + first_q_tag_end + 1)]
        tag_removed = trimmed.replace(tag, '')
        end_tag_begin = tag_removed[(first_q_tag_begin - 1):].find("</a>")
        if end_tag_begin > 0:
            result = tag_removed.replace("</a>", '')
            return result
        return tag_removed
    return trimmed


def render_page(title, body_soup) -> str:
    body = '\n'.join([str(item) for item in body_soup])
    return HTML_TEMPLATE_FORMAT_STRING.format(title, body)


def tags_to_indicies(soup, tags: List[object]) -> List[int]:
    all_ = soup.find_all()
    return [all_.index(tag) for tag in tags]


def elminate_named_duplicates(tags):
    safe = []
    for tag in tags:
        tag_name = str(tag["name"])
        if tag_name in safe:
            tags.remove(tag)
        else:
            safe.append(tag_name)


def balance_name_tag(tags):
    for tag in tags:
        if not tag.get('name'):
            tag['name'] = tag['id']
        elif not tag.get('id'):
            tag['id'] = tag['name']


def question_tags(q_body, letter_part) \
        -> Tuple[List[int], List[int], List[str]]:
    a_regex = re.compile(f"{letter_part}\\d+")
    answer_begins = q_body.find_all(
                    lambda tag: (tag.name == 'a' and (a_regex.search(tag.get("name") or "")
                                or a_regex.search(tag.get("id") or ""))) if tag else False
                )
    balance_name_tag(answer_begins)
    elminate_named_duplicates(answer_begins)
    names = [answer["name"] for answer in answer_begins]
    answer_ends = q_body.find_all('a', href=re.compile(
        f"Answers{letter_part}\\d+/answer{letter_part}\\d+.html"))
    begin_indicies = tags_to_indicies(q_body, answer_begins)
    end_indicies = tags_to_indicies(q_body, answer_ends)
    return begin_indicies, end_indicies, names


def extract_betweens(soup, begins_indicies, end_indicies) -> List:
    assert len(begins_indicies) == len(end_indicies)
    all_ = soup.find_all()
    return [
            (all_[begins_indicies[ii]:end_indicies[ii]], )
            for ii in range(len(begins_indicies))
        ]


def qa_urls(root_url, letter_part, begin, end) -> Tuple[str, List[str]]:
    question_url = f"{root_url}/Part{letter_part}/Cpuzzles{letter_part}section{begin:02}.html"
    answer_urls = [
            f"{root_url}/Part{letter_part}/Answers{letter_part}{begin:02}/answer{letter_part}{ii:02}.html"
            for ii in range(begin, end + 1)
        ]
    return question_url, answer_urls


def html_to_pdf(path):
    options = {"enable-local-file-access": "", 'quiet': ''}
    html_path = str(path.parent / (path.stem + ".html"))
    pdf_path = str(path.parent / (path.stem + ".pdf"))
    pdfkit.from_file(html_path, pdf_path, options=options)


def html_str_to_pdf(path, html_string):
    options = {"enable-local-file-access": "", 'quiet': ''}
    pdf_path = str(path.parent / (path.stem + ".pdf"))
    pdfkit.from_string(html_string, pdf_path, options=options)


@dataclass
class QASegmentPages:
    letter_label: str
    title: str
    top: str
    qs: List[str]
    ans: List[str]
    seg_num: int

    def write_html_to_disk(self, root_path, allow_skip=False):
        seg_num = str(self.seg_num)
        html_root_path = root_path / self.letter_label / seg_num
        html_path = html_root_path / "top.html"
        if not html_path.parent.exists():
            html_path.parent.mkdir(parents=True, exist_ok=True)
        if not allow_skip and not html_path.exists():
            with open(html_path, 'w') as fs:
                fs.write(self.top)
        for qi, q in enumerate(self.qs):
            html_path = html_root_path / f"{qi + 1}q.html"
            if allow_skip and html_path.exists():
                continue
            with open(html_path, 'w') as fs:
                fs.write(q)
        for ai, a in enumerate(self.ans):
            if allow_skip and html_path.exists():
                continue
            html_path = html_root_path / f"{ai + 1}a.html"
            with open(html_path, 'w') as fs:
                fs.write(a)

    def convert_to_pdf(self, root_path, allow_skip=False):
        seg_num = str(self.seg_num)
        html_root_path = root_path / self.letter_label / seg_num
        path = html_root_path / "top.html"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        html_to_pdf(path)
        pool = concurrent.futures.ThreadPoolExecutor(
                thread_name_prefix="book-pdf-out")
        for qi, q in enumerate(self.qs):
            path = html_root_path / f"{qi + 1}q"
            if path.exists() and allow_skip:
                continue
            pool.submit(lambda p: html_to_pdf(p), copy.copy(path))
            # html_to_pdf(path)
        for ai, a in enumerate(self.ans):
            if path.exists() and allow_skip:
                continue
            path = html_root_path / f"{ai + 1}a"
            pool.submit(lambda p: html_to_pdf(p), copy.copy(path))
            # html_to_pdf(path)
        pool.shutdown(wait=True)


def extract_questions(q_soup, letter_part) -> List[str]:
    answer_begin_indicies, answer_end_indicies, names = question_tags(
            q_soup, letter_part)
    questions = extract_betweens(
            q_soup, answer_begin_indicies, answer_end_indicies)
    return [
        bold_codes_syntax(render_page(names[ii], *elements), C_KEYWORDS)
        for ii, elements in enumerate(questions)
    ]


def process_question_set(q_soup, a_soups, letter_part, seg_number) -> QASegmentPages:
    q_top, q_title = extract_body(q_soup)
    q_top = q_top_page(q_soup, letter_part, seg_number)
    answer_body_titles = [extract_body(a) for a in a_soups]
    answers = [
        bold_codes_syntax(render_page(at, ab), C_KEYWORDS)
        for ab, at in answer_body_titles
    ]
    questions = extract_questions(q_soup, letter_part)
    return QASegmentPages(
            letter_part,
            q_title,
            bold_codes_syntax(
                HTML_TEMPLATE_FORMAT_STRING.format(q_title, q_top),
                C_KEYWORDS
            ),
            list(questions),
            list(answers),
            seg_number
        )


book_index = {
    "A": 4,
    "B": [1, 15],
    "C": 5,
    "D": 2,
    "E": 4,
    "F": 3,
    "G": 3,
    "S": 2
}


chapter_titles = {
    "A": "Loops",
    "B": "Random Numbers",
    "C": "1D Arrays",
    "D": "2D Arrays",
    "E": "Scope and Linkage",
    "F": "Pointers",
    "G": "Structs",
    "S": "Strings"
}


special_index = {
    "S": [(0, "stringsIntro.pdf", "Intro To Strings")]
}


def download_images(local_root, page_url, soup, letter_part, seg_num, sleep_time=5):
    image_urls = [page_url.parent / image["src"] for image in soup.find_all("img")]
    for url in image_urls:
        image_path = local_root / letter_part / str(seg_num) / (url.stem + url.suffix)
        if image_path.exists():
            continue
        elif not image_path.parent.exists():
            image_path.parent.mkdir(parents=True)
        time.sleep(sleep_time + random.randrange(1, 10) / 10)
        image = requests.get(httpsify(url)).content
        print(letter_part, seg_num, url)
        with open(image_path, 'wb') as fs:
            fs.write(image)


def get_all_qa_urls(root_url, book_index) -> Tuple[List[Path], List[Path]]:
    question_urls = []
    answer_urls = []
    for part, segments in book_index.items():
        if type(segments) is list:
            question_url, answer_url = qa_urls(
                    root_url, part, segments[0], segments[1])
            question_urls.append(question_url)
            answer_urls += answer_url
        else:
            for ii in range(segments):
                question_url, answer_url = qa_urls(
                        root_url, part, ii * 10 + 1, (ii + 1) * 10)
                question_urls.append(question_url)
                answer_urls += answer_url
    return question_urls, answer_urls


def download_page(url, local_source: Path = LOCAL_SOURCE,
                  parent_folder=True, allow_skip=True) -> bool:
    url = Path(url)
    local_path = (local_source / url.parent.stem) \
        if parent_folder else local_source
    if not local_path.exists():
        local_path.mkdir(exist_ok=True, parents=True)
    page_path = local_path / f"{url.stem}.html"
    if page_path.exists():
        return False
    q_page = requests.get(httpsify(url)).content
    with open(page_path, 'w') as fs:
        q_page = q_page.replace(b'\x97', '-'.encode('utf-8'))
        fs.write(q_page.decode('utf-8'))
    return True


def download_qa_contents(q_urls, a_urls, status=True, delay=5, allow_skip=True, 
                         local_source: Path = LOCAL_SOURCE):
    do_delay = False
    for ii in tqdm.tqdm(range(len(q_urls)), desc="Downloading Questions"):
        qu = q_urls[ii]
        if do_delay:
            time.sleep(delay + random.randrange(1, 10) / 10)
        if status:
            print(qu)
        do_delay = download_page(qu, allow_skip=allow_skip, local_source=local_source)
    for ii in tqdm.tqdm(range(len(a_urls)), desc="Downloading Answers"):
        au = a_urls[ii]
        if do_delay:
            time.sleep(delay + random.randrange(1, 10) / 10)
        if status:
            print(au)
        do_delay = download_page(au, allow_skip=allow_skip, local_source=local_source)


def make_qa_soups(q_urls, a_urls) \
        -> Tuple[List[BeautifulSoup], List[List[BeautifulSoup]]]:
    q_soups = [
        BeautifulSoup(requests.get(qu).content, PARSER)
        for qu in q_urls
    ]
    a_soups = [[
        BeautifulSoup(requests.get(au).content, PARSER)
        for au in aus
    ] for aus in a_urls]
    return q_soups, a_soups


def make_qa_soups_from_files(q_urls, a_urls) \
        -> Tuple[List[BeautifulSoup], List[List[BeautifulSoup]]]:
    q_soups = []
    qbar = tqdm.tqdm(len(q_urls))
    for qu in q_urls:
        qbar.refresh()
        qbar.update()
        with open(qu, 'r') as fs:
            data = fs.read().replace("â€”", "'")
            q_soups.append(BeautifulSoup(data, PARSER))
    a_soups = []
    a_bar = tqdm.tqdm(len(a_urls))
    for au in a_urls:
        a_bar.refresh()
        a_bar.update()
        with open(au, 'r') as fs:
            data = fs.read()
            a_soups.append(BeautifulSoup(data, PARSER))
    return q_soups, a_soups


def download_all_images(root_url=ROOT_URL, local_root: Path = LOCAL_ROOT):
    qus, aus = get_all_qa_urls(root_url, book_index)
    qusl, ausl = get_all_qa_urls(str(local_root), book_index)
    print("Making Soup")
    ques, answs = make_qa_soups_from_files([], ausl)
    print("Soup Made")
    q_index = 0
    a_range = 0
    a_index = 0
    for letter, section_count in book_index.items():
        print(letter)
        if type(section_count) is list:
            a_range = section_count[1]
            section_count = 1
        else:
            a_range  = 10 * section_count
        print(f"{letter}q")
        for ii in tqdm.tqdm(range(section_count)):
            download_images(local_root,
                            Path(qus[q_index]), ques[q_index], letter, ii + 1)
            q_index += 1
        for ii in tqdm.tqdm(range(section_count)):
            download_images(local_root, Path(aus[a_index]),
                            answs[a_index][ii], letter, ii + 1)
            a_index += 1
    print("Done")


def download_qa(root_url=ROOT_URL, allow_skip=True):
    qus, aus = get_all_qa_urls(root_url, book_index)
    download_qa_contents(qus, aus, allow_skip=allow_skip)


def generate_qa_segments(local_source: Path = LOCAL_SOURCE,
                         allow_skip=False, book_index=book_index):
    qus, aus = get_all_qa_urls(local_source, book_index)
    print("Making Soup")
    ques, answs = make_qa_soups_from_files(qus, aus)
    print("Soup Made")
    a_index = 0
    q_index = 0
    assert len(aus) == len(answs)
    for letter, section_count in book_index.items():
        if type(section_count) is list:
            a_range = section_count[1]
            section_count = 1
        else:
            a_range = 10
        for ii in tqdm.tqdm(range(1, section_count + 1), desc=f"Section {letter}: "):
            a_rng = slice(a_index, (a_index + a_range))
            seg = process_question_set(
                    copy.deepcopy(ques[q_index]),
                    copy.deepcopy(answs[a_rng]),
                    letter,
                    ii
                )
            seg.write_html_to_disk(Path(".") / "local" / "out", allow_skip=allow_skip)
            seg.convert_to_pdf(Path(".") / "local" / "out", allow_skip=allow_skip)
            del seg
            a_index += a_range
            q_index += 1


def download_special(local_root, url, letter_part,
                     before_or_after: str | int = 'b',
                     allow_skip: bool = True) -> bool:
    seg_num = 0 if before_or_after == 'b' else before_or_after + 1
    page_path = local_root / letter_part / str(seg_num) / (url.stem + url.suffix)
    if page_path.exists() and allow_skip:
        return False
    special = requests.get(httpsify(url)).content
    special_soup = BeautifulSoup(special, PARSER)
    page, title = extract_body(special_soup)
    special_soup = download_images(
            local_root, url, page, letter_part, seg_num
        )
    with open(page_path, 'w') as fs:
        fs.write(render_page(title, page))
    return True


def download_all_special(
        local_root: Path = LOCAL_ROOT,
        root_url: Path = ROOT_URL,
        index: Dict[str, int | List[int]] = book_index,
        allow_skip: bool = False):
    if "S" in index:
        special_url = Path(root_url) / "PartS" / "stringsIntro.html"
        download_special(local_root, special_url, 'S', 'b', allow_skip)
        html_to_pdf(local_root / 'S' / '0' / "stringsIntro")


def flip_pages(pages: pypdf.PageRange):
    for page in pages:
        page.rotate(180)


def add_pages(from_: PdfReader, to: PdfWriter,
              title: str = None, parent=None, reveresed: bool = False):
    first_page = None
    for page in from_.pages:
        if first_page is None:
            first_page = len(to.pages)
        to.add_page(page)
    outline_object = None
    if title is not None:
        outline_object = to.add_outline_item(
            title,
            page_number=first_page,
            parent=parent
        )
    return first_page, outline_object


def add_qa_segment(
                local_root: Path,
                book: PdfWriter,
                letter_part: str,
                seg_num: int,
                parent_outline
        ):
    qa_root = local_root / letter_part / str(seg_num)
    intro = PdfReader(qa_root / "top.pdf")
    _, segment_root_outline = add_pages(
            intro,
            book,
            f"Chapter {letter_part} Part {seg_num}",
            parent_outline
        )
    answer_paths = glob.glob(str(qa_root / "*a.pdf"))
    question_paths = glob.glob(str(qa_root / "*q.pdf"))
    answer_paths = list(sorted(answer_paths))
    question_paths = list(sorted(question_paths))
    assert len(answer_paths) == len(question_paths)
    question_count = 0
    for question_path, answer_path in zip(question_paths, answer_paths):
        question_count += 1
        question = PdfReader(question_path)
        question_title = \
            f"Question {letter_part}{((seg_num - 1)) * 10 + question_count}"
        add_pages(
                question,
                book,
                question_title,
                segment_root_outline
            )
        answer = PdfReader(answer_path)
        flip_pages(answer.pages)
        add_pages(
                answer,
                book,
                f"Answer {letter_part}{((seg_num - 1)) * 10 + question_count}",
                segment_root_outline,
                reveresed = True
            )
    return segment_root_outline


def make_chapter_pages(
            local_root: Path = LOCAL_ROOT,
            book_index: Dict[str, int | List[int]] = book_index,
            chapter_titles: Dict[str, str] = chapter_titles
        ):
    for letter, seg_count in book_index.items():
        html_code = "<html><head></head><body>" \
                + "<center style='font-size:xx-large'><strong>Chapter " \
                + f"{letter}: {chapter_titles[letter]}</center></strong>" \
                + "</body></html>"
        html_str_to_pdf(local_root / f"chap{letter}", html_code)


def generate_index_structure(
                index: Dict[str, int],
                chapters: PdfReader,
                item_list: List,
                depth: int
            ) -> Dict:
    for item in item_list:
        if type(item) is list:
            generate_index_structure(index, chapters, item, depth + 1)
        else:
            title = item["/Title"]
            page_number = chapters.get_destination_page_number(item)
            index[title] = [depth, page_number]


def generate_index_html(
                    chapters: PdfReader,
                    index: Dict,
                    max_page_order: int,
                    book_index: Dict = book_index,
                    depth: int = 0,
                    tab_width: int = 4,
                    tab_char: str = '-',
                    line_length: int = 120,
                    dot_char: str = '.',
                    preamble_size: int = 4
               ) -> str:
    tab = tab_width * tab_char
    html = "<html><title>Index</title><head><center style='font-size: xx-large'>Index</center></head><body style='font-family: monospace'><center>"
    progress = tqdm.tqdm(len(index), desc="Generating HTML")
    reg = re.compile("Chapter ([A-Z]) Part (\\d)")
    q_reg = re.compile("Question.+")
    a_reg = re.compile("Answer.+")
    # ostr = '0' + str(max_page_order)
    for title, item in index.items():
        q_match = q_reg.search(title)
        a_match = a_reg.search(title)
        if a_match or q_match:
            continue
        line = f"\t<p >{tab * item[0]}{title}"
        reg_match = reg.match(title)
        if reg_match is not None:
            letter = reg_match.group(1)
            seg_count = book_index[letter]
            if type(seg_count) is list:
                seg_count = book_index[letter][1]
            else:
                seg_count = 10
            seg_num = reg_match.group(2)
            seg_num = int(seg_num)
            seg_begin = ((seg_num - 1) * seg_count + 1)
            seg_end = (seg_num * seg_count)
            qas = f": <i>Questions & Answers {letter}{seg_begin:03}..{letter}{seg_end:03}</i>"
            line = line + qas
        line += f"{dot_char * (line_length - len(line))}{(item[1] + preamble_size):03}</p>\n"
        html += line
        progress.update()
    html += "</center></body></html>"
    return html


def generate_index(
            title: str = C_PUZZLERS,
            local_root: Path = LOCAL_ROOT,
            book_index: Dict[str, int | List[int]] = book_index,
            special_index: Dict[str, List[str]] = special_index,
            chapter_titles: Dict[str, str] = chapter_titles
        ):
    chapters = PdfReader(local_root / f"{title} Chapters.pdf")
    index = {}
    generate_index_structure(index, chapters, chapters.outline, 0)
    max_page_order = math.log10(len(chapters.pages))
    html = generate_index_html(chapters, index, max_page_order)
    index_html_path = local_root / f"{title} Index.html"
    index_path = local_root / f"{title} Index"
    with open(index_html_path, 'w') as fs:
        fs.write(html)
    html_to_pdf(index_path)


def assemble_preamble(
            title: str = C_PUZZLERS,
            local_root: Path = LOCAL_ROOT,
            book_index: Dict[str, int | List[int]] = book_index,
            special_index: Dict[str, List[str]] = special_index,
            chapter_titles: Dict[str, str] = chapter_titles
        ):
    book = PdfWriter()
    cover = PdfReader(local_root / "Cover.pdf")
    index = PdfReader(local_root / f"{title} Index.pdf")
    for page in cover.pages:
        book.add_page(page)
    book.add_outline_item("Cover", page_number=0)
    book.add_outline_item("Forward", page_number=1)
    for page in index.pages:
        book.add_page(page)
    book.add_outline_item("Index", page_number=2)
    with open(local_root / f"{title} Preamble.pdf", 'wb') as fs:
        book.write(fs)


def assemble_chapters(
            title: str = C_PUZZLERS,
            local_root: Path = LOCAL_ROOT,
            book_index: Dict[str, int | List[int]] = book_index,
            special_index: Dict[str, List[str]] = special_index,
            chapter_titles: Dict[str, str] = chapter_titles
        ):
    book = PdfWriter()
    for letter, seg_count in book_index.items():
        if type(seg_count) is list:
            seg_count = 1
        chap = PdfReader(local_root / f"chap{letter}.pdf")
        book.add_page(chap.pages[-1])
        page_num = len(book.pages) - 1
        chapter_root = book.add_outline_item(
                f"Chapter {letter}: {chapter_titles[letter]}", page_num)
        if letter in special_index:
            for folder, special_file, special_title in special_index[letter]:
                special_path = local_root / letter / '0' / special_file
                special = PdfReader(str(special_path))
                first_page = len(book.pages)
                for page in special.pages:
                    book.add_page(page)
                book.add_outline_item(
                         special_title,
                         page_number=first_page,
                         parent=chapter_root
                     )
        for seg_num in tqdm.tqdm(range(1, seg_count + 1),
                                 desc=f"Chapter {letter} Segments"):
            add_qa_segment(
                        local_root,
                        book,
                        letter,
                        seg_num,
                        chapter_root
                    )
    with open(local_root / f"{title} Chapters.pdf", 'wb') as fs:
        book.write(fs)


def assemble_book(
            title: str = C_PUZZLERS,
            local_root: Path = LOCAL_ROOT,
            book_index: Dict[str, int | List[int]] = book_index,
            special_index: Dict[str, List[str]] = special_index,
            chapter_titles: Dict[str, str] = chapter_titles
        ):
    book = PdfWriter()
    preamble = open(local_root / f"{title} Preamble.pdf", 'rb')
    chapters = open(local_root / f"{title} Chapters.pdf", 'rb')
    book.merge(position=1, fileobj=preamble)
    book.merge(position=3, fileobj=chapters)
    with open(local_root / f"{title}.pdf", 'wb') as fs:
        book.write(fs)
    book.close()


def assemble_book_contents(
            title: str = C_PUZZLERS,
            local_root: Path = LOCAL_ROOT,
            book_index: Dict[str, int | List[int]] = book_index,
            special_index: Dict[str, List[str]] = special_index,
            chapter_titles: Dict[str, str] = chapter_titles
        ):
    assemble_preamble(title, local_root, book_index,
                      special_index, chapter_titles)
    assemble_chapters(title, local_root, book_index,
                      special_index, chapter_titles)
    assemble_book(title, local_root, book_index,
                      special_index, chapter_titles)


def dir_path(string): # Credit to SO question 38834378
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("command")
    argparser.add_argument("--allow-skip", '-k', action='store_true')
    argparser.add_argument("--section", '-S', type=str)
    argparser.add_argument("--segment-number", '-s', type=int)
    argparser.add_argument(
            "--local-root",
            '-r',
            type=Path,
            default=LOCAL_ROOT
        )
    arguments = argparser.parse_args()
    local_root = Path(arguments.local_root)
    allow_skip = arguments.allow_skip if arguments.allow_skip else False
    if arguments.command == "download:special":
        download_all_special(local_root=local_root, allow_skip=allow_skip)
    elif arguments.command == "download:images":
        download_all_images(local_root=local_root)
    elif arguments.command == "download:qa":
        download_qa(allow_skip)
    elif arguments.command == "generate:chapter_pages":
        make_chapter_pages(local_root, book_index, chapter_titles)
    elif arguments.command == "generate:index":
        generate_index(local_root=local_root)
    elif arguments.command == "generate:qa_segments":
        if arguments.section:
            generate_qa_segments(LOCAL_SOURCE, allow_skip, {
                arguments.section: book_index[arguments.section]})
        else:
            generate_qa_segments(LOCAL_SOURCE, allow_skip)
    elif arguments.command == "assemble:segment":
        segment = PdfWriter()
        add_qa_segment(
                    local_root,
                    segment,
                    arguments.section,
                    arguments.segment_number,
                    None
                )
        segment_path = local_root / arguments.section \
            / str(arguments.segment_number) / "segment.pdf"
        with open(segment_path, 'wb') as fs:
            segment.write(fs)
    elif arguments.command == "assemble:book":
        assemble_book(local_root=local_root)
    elif arguments.command == "assemble:book_contents":
        assemble_book_contents(local_root=local_root)
    elif arguments.command == "assemble:preamble":
        assemble_preamble(local_root=local_root)
    elif arguments.command == "assemble:chapters":
        assemble_chapters(local_root=local_root)
    else:
        print("NOOP: Unrecognized Command")
        sys.exit(1)


if __name__ == "__main__":
    main()
