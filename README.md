# JacRED KG Extraction: Mention-Span Constrained Entity Normalization

日本語文書レベル関係抽出データセット **JacRED** を用いた、低コストLLM（Gemini Flash系）による知識グラフトリプル抽出の実験リポジトリ。本リポジトリでは **Mention-Span Constrained Entity Normalization** 手法を実装し、Baselineとの比較実験を行う。

---

## 目次

1. [概要](#1-概要)
2. [背景・動機](#2-背景動機)
3. [データセット](#3-データセット)
4. [実験設定](#4-実験設定)
5. [手法](#5-手法)
6. [結果](#6-結果)
7. [分析](#7-分析)
8. [再現方法](#8-再現方法)
9. [ファイル構成](#9-ファイル構成)
10. [参考文献](#10-参考文献)

---

## 1. 概要

本プロジェクトは、日本語文書群からエンティティ（固有表現）とエンティティ間の関係（relation）を抽出し、知識グラフ（Knowledge Graph）のトリプル `(head_entity, relation, tail_entity)` を自動構築する手法の実験である。

Google Gemini Flash系モデルの **Structured Outputs**（JSON Schema強制出力）を活用し、以下の2条件を比較する:

- **Baseline（One-shot抽出）**: 1回のLLM呼び出しでエンティティと関係を同時に抽出
- **Mention-Span Constrained Entity Normalization（EntityNorm）**: エンティティメンションのスパン抽出 → canonical_nameによるクラスタリング → クラスタ化エンティティを用いた関係抽出 → domain/range型制約による後処理

EntityNorm手法の核心は、LLMに文書中の**正確なテキストスパン（verbatimな部分文字列）** を抽出させ、同一エンティティへの異なる言及（共参照）を**canonical_name（正規化名）** でグループ化した上で関係を抽出する点にある。これにより、JacREDのvertexSet構造（1つのエンティティが複数のmentionを持つ）に近い表現が得られ、エンティティアライメントの精度向上が期待される。

評価はJacRED devセットから選択した10文書に対して、文書レベル関係抽出の標準指標（Precision / Recall / F1）で行う。

---

## 2. 背景・動機

### 2.1 文書レベル関係抽出（Document-level Relation Extraction, DocRE）

文書レベル関係抽出（DocRE）は、1つの文書全体を入力として、文書中に出現するエンティティペア間の関係を全て抽出するタスクである。文単位の関係抽出（Sentence-level RE）とは異なり、複数文にまたがる推論や共参照解析が必要となる。

具体的には以下の処理を行う:
1. 文書中のエンティティ（人名、組織名、地名など）を認識する
2. 全てのエンティティペア `(head, tail)` について、既定の関係タイプ集合から該当する関係を判定する
3. 関係が存在しないペアには "NA"（関係なし）を割り当てる

### 2.2 JacREDデータセット

**JacRED**（Japanese Document-level Relation Extraction Dataset）は、英語DocREデータセット **DocRED**（Yao et al., ACL 2019）の構造を日本語Wikipedia記事に適用して構築されたデータセットである。Ma et al.（LREC-COLING 2024）が、cross-lingual transferを活用したアノテーション支援手法により作成した。

### 2.3 本実験の目的

1. **低コストLLMの有効性検証**: 高価なGPT-4系モデルではなく、Gemini Flash系（低コスト・高速）モデルでDocREがどの程度可能かを検証する
2. **Structured Outputsの活用**: 自由形式テキスト出力ではなくJSON Schema強制出力を用い、パースエラーや不正出力を排除する
3. **エンティティ正規化の有効性**: Mention-Span Constrained Entity Normalization手法により、エンティティアライメント精度を向上させ、結果としてPrecision/F1の改善が可能かを測定する
4. **共参照解決の効果**: LLMにスパンレベルの抽出とcanonical_nameの統一を行わせることで、JacREDのvertexSet構造により忠実なエンティティ表現が得られるかを検証する
5. **最終応用**: 日本語文書コレクションからの大規模知識グラフ自動構築への基盤技術確立

---

## 3. データセット

### 3.1 JacRED概要

| 項目 | 内容 |
|---|---|
| 名称 | JacRED (Japanese Document-level Relation Extraction Dataset) |
| ソース | https://github.com/YoumiMa/JacRED |
| 論文 | Ma et al., "Building a Japanese Document-Level Relation Extraction Dataset Assisted by Cross-Lingual Transfer", LREC-COLING 2024 |
| 言語 | 日本語（Wikipedia記事由来） |
| ベース | DocRED（Yao et al., ACL 2019）の構造を日本語に適用 |

### 3.2 データ分割

| 分割 | 文書数 | 用途 |
|---|---|---|
| train | 1,400 | 訓練（本実験ではfew-shot例選択とdomain/range制約テーブル構築に使用） |
| dev | 300 | 開発・評価（本実験では10文書を選択して評価に使用） |
| test | 300 | テスト（本実験では未使用） |

各分割間に文書の重複はない。

### 3.3 データフォーマット

各文書は以下のフィールドを持つJSONオブジェクトである:

```json
{
  "title": "文書タイトル（Wikipedia記事名）",
  "sents": [
    ["トークン1", "トークン2", "..."],
    ["トークン1", "トークン2", "..."]
  ],
  "vertexSet": [
    [
      {"name": "エンティティ名", "type": "PER", "sent_id": 0, "pos": [3, 5]}
    ]
  ],
  "labels": [
    {"h": 0, "t": 1, "r": "P27", "evidence": [0, 2]}
  ]
}
```

**各フィールドの説明:**

- **`title`**: Wikipedia記事のタイトル文字列
- **`sents`**: トークン化済みの文のリスト。各文はトークン（文字列）のリスト。元のテキストは各文のトークンを結合（join）して再構成する
- **`vertexSet`**: エンティティのリスト。各エンティティは1つ以上の **mention**（言及）を持つ。各mentionは:
  - `name`: 言及テキスト（例: "東京都"）
  - `type`: エンティティタイプ（後述の9種類のいずれか）
  - `sent_id`: この言及が出現する文のインデックス（0始まり）
  - `pos`: 文中のトークン位置 `[start, end)`（半開区間）
- **`labels`**: 正解関係ラベルのリスト。各ラベルは:
  - `h`: headエンティティのvertexSetインデックス（0始まり）
  - `t`: tailエンティティのvertexSetインデックス（0始まり）
  - `r`: 関係タイプのPコード（例: "P27"）
  - `evidence`: 根拠となる文のインデックスリスト

### 3.4 エンティティタイプ（9種類）

| タイプコード | 日本語説明 | 例 |
|---|---|---|
| `PER` | 人物 | 織田信長、アインシュタイン |
| `ORG` | 組織 | トヨタ自動車、国連 |
| `LOC` | 場所・地名 | 東京都、ナイル川 |
| `ART` | 作品・人工物・賞 | あずきちゃん、ノーベル賞 |
| `DAT` | 日付 | 1964年5月12日、2011年9月 |
| `TIM` | 時間 | 午前10時 |
| `MON` | 金額 | 100万円 |
| `%` | パーセンテージ・数値 | 50%、3.14 |
| `NA` | 該当なし（未分類） | --- |

注: 本実験のLLMプロンプトでは `NA` を除く8種類をエンティティタイプとして指定する。Structured Outputsのスキーマ（`schemas.py`の`EXTRACTION_SCHEMA`および`SPAN_ENTITY_SCHEMA`）では `enum: ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]` として8タイプに制限している。

### 3.5 関係タイプ（35種類）

以下はJacREDで定義される全35種類の関係タイプである。各行はWikidataのプロパティコード（Pコード）、英語名、日本語説明を示す。日本語説明は本実験のLLMプロンプト（`prompts.py`の`RELATION_JAPANESE`辞書）で使用されるものと同一である。

| Pコード | English Name | 日本語説明 |
|---|---|---|
| `P1376` | capital of | 首都（〜の首都である） |
| `P131` | located in the administrative territorial entity | 行政区画（〜に位置する行政区画） |
| `P276` | location | 所在地（〜に所在する） |
| `P937` | work location | 活動場所（〜で活動した） |
| `P27` | country of citizenship | 国籍（〜の国籍を持つ） |
| `P569` | date of birth | 生年月日 |
| `P570` | date of death | 没年月日 |
| `P19` | place of birth | 出生地 |
| `P20` | place of death | 死没地 |
| `P155` | follows | 前任・前作（〜の後に続く） |
| `P40` | child | 子（〜の子である） |
| `P3373` | sibling | 兄弟姉妹 |
| `P26` | spouse | 配偶者 |
| `P1344` | participant in | 参加イベント（〜に参加した） |
| `P463` | member of | 所属（〜に所属する） |
| `P361` | part of | 上位概念（〜の一部である） |
| `P6` | head of government | 首長（〜の首長である） |
| `P127` | owned by | 所有者（〜に所有される） |
| `P112` | founded by | 設立者（〜が設立した） |
| `P108` | employer | 雇用主（〜に雇用される） |
| `P137` | operator | 運営者（〜が運営する） |
| `P69` | educated at | 出身校（〜で教育を受けた） |
| `P166` | award received | 受賞（〜を受賞した） |
| `P170` | creator | 制作者（〜が制作した） |
| `P175` | performer | 出演者・パフォーマー |
| `P123` | publisher | 出版社（〜が出版した） |
| `P1441` | present in work | 登場作品（〜に登場する） |
| `P400` | platform | プラットフォーム |
| `P36` | capital | 首都（〜が首都である） |
| `P156` | followed by | 後任・次作（〜の前にある） |
| `P710` | participant | 参加者（〜が参加した） |
| `P527` | has part | 構成要素（〜を含む） |
| `P1830` | owner of | 所有物（〜を所有する） |
| `P121` | item operated | 運営対象（〜を運営する） |
| `P674` | characters | 登場人物（作品の登場人物） |

注: `P1376`（capital of）と`P36`（capital）、`P155`（follows）と`P156`（followed by）、`P361`（part of）と`P527`（has part）、`P127`（owned by）と`P1830`（owner of）、`P137`（operator）と`P121`（item operated）はそれぞれ逆方向の関係ペアである。

### 3.6 データセット統計（参考値）

| 指標 | 値（概算） |
|---|---|
| 平均エンティティ数/文書 | 約17 |
| 平均関係数/文書 | 約20 |
| 平均トークン数/文書 | 約253 |
| 関係密度（関係数 / 可能なペア数） | 約6.5% |

---

## 4. 実験設定

### 4.1 文書選択

#### 評価文書（10文書）

JacRED devセット（300文書）から、文書の文字数（`char_count`）でソートし、等間隔で10文書を選択する **層化サンプリング** を行った。具体的には:

```python
sorted_docs = sorted(dev_data, key=char_count)  # 文字数昇順ソート
total = len(sorted_docs)  # 300
indices = [int(total * (i + 0.5) / 10) for i in range(10)]
# indices = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285]
```

これにより、短い文書から長い文書まで均等にカバーする10文書が選ばれる。

選択された10文書:

| # | タイトル | Gold エンティティ数 | Gold 関係数 |
|---|---|---|---|
| 1 | ダニエル・ウールフォール | 9 | 12 |
| 2 | アンソニー世界を駆ける | 9 | 6 |
| 3 | 青ナイル州 | 11 | 20 |
| 4 | 小谷建仁 | 17 | 11 |
| 5 | 窪田僚 | 18 | 11 |
| 6 | イーオー | 17 | 10 |
| 7 | 堂山鉄橋 | 15 | 9 |
| 8 | 木村千歌 | 21 | 25 |
| 9 | バハン地区 | 18 | 28 |
| 10 | ジョー・ギブス | 31 | 16 |

合計: Gold関係数 = 148

#### Few-shot例（1文書）

**Few-shot prompting（少数例プロンプティング）** とは、LLMに対してタスクの入出力例を少数（1〜数個）プロンプト中に含めることで、タスクの期待フォーマットや振る舞いをモデルに示す手法である。例を0個与える場合を **zero-shot**、1個の場合を **one-shot**、複数個の場合を **few-shot** と呼ぶ。

本実験では **1例（one-shot）** を採用した。これは以下のトレードオフに基づく判断である:
- **コスト・コンテキスト長の制約**: 例を増やすと入力トークン数が増加し、APIコストが上昇する。また、モデルのコンテキストウィンドウ（入力可能なトークン数の上限）を圧迫し、対象文書の処理に使えるトークン数が減少する。
- **品質向上の限界**: 1例でタスク形式を十分に示せる場合、2例以上に増やしても品質改善は限定的であることが多い。
- **例の品質**: 使用する例は**訓練データのGold label（正解ラベル）** から構成される。つまり、人手でアノテーションされた正しい入出力ペアをモデルに見せることで、期待される出力形式と粒度を正確に伝えている。

訓練データから以下の条件を満たす文書を1つ選択する:

- 文字数: 150 - 250文字
- エンティティ数: 5 - 12
- 関係ラベル数: 3 - 15

条件を満たす候補のうち最も短いものを選択する。

選択されたfew-shot文書: **「スタッド (バンド)」**

### 4.2 モデル構成

本実験では以下の1つのモデル構成を使用した:

| 構成名 | モデルID | thinking_budget | 説明 |
|---|---|---|---|
| gemini-3-flash-preview (ON) | `gemini-3-flash-preview` | 2048 | `ThinkingConfig(thinking_budget=2048)` を指定。モデルが最大2048トークンの内部推論を行ってから最終回答を生成 |

#### "thinking"（思考モード）とは何か

Gemini 2.5 Flash以降のモデルは **thinking**（内部推論 / 思考モード）機能を持つ。これは **chain-of-thought reasoning（連鎖的思考推論）** をモデル内部に組み込んだ機能であり、従来のプロンプトエンジニアリングで「ステップバイステップで考えてください」と指示する手法（chain-of-thought prompting）とは異なり、モデルのアーキテクチャレベルで推論プロセスが組み込まれている。

**動作原理:**

thinking が有効な場合、モデルはユーザのリクエストに対して以下の2段階で応答を生成する:

1. **内部推論フェーズ（thinking tokens）**: モデルはまず「推論トークン」（reasoning tokens / thinking tokens）を内部的に生成する。これはモデルが問題を分解し、ステップバイステップで考えるためのトークン列である。重要な点として、**これらの推論トークンはAPIレスポンスのテキストには含まれない**（呼び出し側からは見えない）。つまり、ユーザが受け取る最終出力には推論過程は表示されず、結論のみが返される。
2. **最終回答生成フェーズ**: 内部推論が完了した後、モデルはその推論結果を踏まえて最終的な回答を生成する。この回答のみがAPIレスポンスとして返される。

**`thinking_budget` パラメータ:**

`thinking_budget` は `ThinkingConfig` の設定項目で、**モデルが内部推論に使用できるトークンの最大数**を制御する。

- **`thinking_budget=0`**: thinking機能を**完全に無効化**する。モデルは内部推論フェーズをスキップし、通常の（non-reasoning）モデルと同等に動作する。即座に最終回答の生成を開始するため、レイテンシが低い。
- **`thinking_budget=2048`**: モデルが**最大2048トークンの内部推論**を行うことを許可する。ただし、タスクが単純な場合、モデルは2048トークン全てを使い切らず、より少ないトークン数で推論を完了する場合がある（上限であり、必ず使い切るわけではない）。
- **一般的な傾向**: thinking_budgetを大きくすると、(a) レイテンシが増加する（推論トークン生成の時間がかかる）、(b) APIコストが増加する（推論トークンも課金対象として計上される）、(c) 複雑なタスクでは回答品質が向上する可能性がある。

**コードでの指定方法（`llm_client.py`）:**

```python
from google.genai.types import GenerateContentConfig, ThinkingConfig

config = GenerateContentConfig(
    system_instruction=system_prompt,
    response_mime_type="application/json",
    response_schema=response_schema,
    temperature=0.2,
    thinking_config=ThinkingConfig(thinking_budget=2048),
)
```

### 4.3 Structured Outputs（構造化出力）

全てのLLM呼び出しでGoogle GenAI SDKの **Structured Outputs** 機能を使用する。これにより、モデルの出力が指定したJSON Schemaに厳密に従うことが保証される。自由形式テキストの出力やJSONパースエラーは原理的に発生しない。

**Structured Outputsの仕組み:**

通常のLLM呼び出しでは、モデルは自由形式のテキストを生成する。プロンプトで「JSON形式で出力してください」と指示しても、モデルが不正なJSON（閉じ括弧の欠落、余分なテキストの混入など）を出力するリスクがある。Structured Outputsはこの問題を根本的に解決する。

APIリクエストで以下の2つのパラメータを指定する:
- **`response_mime_type="application/json"`**: モデルの出力をJSON形式に強制する。モデルのデコーディングプロセス（トークンを1つずつ選択する過程）において、JSON構文に違反するトークンは選択候補から除外される。これは単なるプロンプト指示ではなく、**デコーディング時のハード制約**（constrained decoding）である。
- **`response_schema=...`**: 出力JSONが準拠すべきJSON Schemaを指定する。スキーマで定義されたフィールド名、型、必須フィールドに違反するトークンはデコーディング時に除外される。

**`enum` 制約の効果:**

本実験では `entities[].type` フィールドに `enum: ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]` を指定している。これにより、モデルはエンティティタイプとしてこの8種類以外の文字列を**物理的に出力できない**。デコーディング時にenum外のトークン列は確率0に設定されるため、「モデルが勝手に新しいタイプを発明する」という問題は原理的に排除される。これはプロンプトで「以下のタイプのみ使用してください」と指示するよりも遥かに信頼性が高い。

#### 抽出用スキーマ（EXTRACTION_SCHEMA）

Baseline・EntityNorm Step 3（関係抽出）の両方で使用する。モデルの出力を以下の構造に強制する:

```json
{
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {
            "type": "string",
            "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]
          }
        },
        "required": ["id", "name", "type"]
      }
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "head": {"type": "string"},
          "relation": {"type": "string"},
          "tail": {"type": "string"},
          "evidence": {"type": "string"}
        },
        "required": ["head", "relation", "tail", "evidence"]
      }
    }
  },
  "required": ["entities", "relations"]
}
```

- `entities[].type` は `enum` 制約により8種類のエンティティタイプのいずれかに強制される
- `relations[].relation` は文字列型だが `enum` 制約はない（後処理で不正なPコードをフィルタする）
- `relations[].head` と `relations[].tail` は `entities[].id` を参照する文字列

#### スパンエンティティ抽出用スキーマ（SPAN_ENTITY_SCHEMA）

EntityNorm Step 1（エンティティメンション抽出）で使用する。モデルの出力を以下の構造に強制する:

```json
{
  "type": "object",
  "properties": {
    "entity_mentions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "mention_text": {"type": "string"},
          "sentence_index": {"type": "integer"},
          "canonical_name": {"type": "string"},
          "type": {
            "type": "string",
            "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]
          }
        },
        "required": ["mention_text", "sentence_index", "canonical_name", "type"]
      }
    }
  },
  "required": ["entity_mentions"]
}
```

- `mention_text`: 文書中のverbatimな部分文字列（正確なスパン）
- `sentence_index`: そのメンションが出現する文の番号（0始まり）
- `canonical_name`: 正規化されたエンティティ名（共参照解決用）
- `type`: エンティティタイプ（8種類のenum制約あり）

### 4.4 API呼び出し設定

**Temperature（温度パラメータ）について:**

LLMがトークン（単語の断片）を1つずつ生成する際、各ステップで次のトークンの確率分布が計算される。**temperature** はこの確率分布の「鋭さ」を制御するパラメータである:
- **temperature=0（またはそれに近い値）**: 確率分布が極端に鋭くなり、最も確率の高いトークンがほぼ確定的に選択される。出力の再現性が高いが、temperature=0では**退化的な繰り返し**（同じフレーズを無限に繰り返す現象）が発生するリスクがある。
- **temperature=1.0**: モデルの学習時の確率分布がそのまま使用される。出力に適度な多様性がある。
- **temperature > 1.0**: 確率分布が平坦化され、低確率のトークンも選択されやすくなる。出力が創造的になるが、不正確な内容が増える。

本実験では **temperature=0.2** を採用した。これは退化的な繰り返しを回避しつつ（temperature=0の問題を避ける）、出力の一貫性と再現性を高める設定である。情報抽出タスクでは創造性より正確性が重要なため、低いtemperatureが適切である。

| パラメータ | 値 | 説明 |
|---|---|---|
| `temperature` | 0.2 | 低めの温度で出力の再現性を高める（上記の解説参照） |
| `max_retries` | 3 | API呼び出し失敗時の最大リトライ回数 |
| リトライ間隔 | 指数バックオフ（2秒、4秒、8秒） | `wait = 2 ** (attempt + 1)` |
| SDK | `google-genai` Python パッケージ | `from google import genai` |
| 認証 | APIキー方式 | 環境変数 `GEMINI_API_KEY` またはファイルから読み込み |

---

## 5. 手法

### 5.1 Baseline: One-shot抽出

1回のLLM呼び出しでエンティティと関係を同時に抽出する。

#### 処理フロー

**Step 1: システムプロンプト構築**

以下の情報を含むシステムプロンプトを構築する:
- タスク説明:「日本語文書から知識グラフ（エンティティと関係）を抽出する」
- エンティティタイプ一覧: 8種類（PER, ORG, LOC, ART, DAT, TIM, MON, %）とその日本語説明
- 関係タイプ一覧: 35種類のPコードと英語名と日本語説明
- ルール: 指定タイプのみ使用、Pコードのみ使用、evidence付与、headとtailにはentities IDを使用

システムプロンプトの実際のテンプレート:
```
あなたは日本語文書から知識グラフ（エンティティと関係）を抽出する専門家です。

## タスク
与えられた日本語文書から、エンティティ（固有表現）とエンティティ間の関係を抽出してください。

## エンティティタイプ（8種類）
  - PER: 人物
  - ORG: 組織
  - LOC: 場所・地名
  - ART: 作品・人工物・賞
  ...（以下略）

## 関係タイプ（35種類、Pコードで指定）
  - P1376 (capital of): 首都（〜の首都である）
  - P131 (located in the administrative territorial entity): 行政区画（〜に位置する行政区画）
  ...（以下略）

## ルール
- エンティティには上記のタイプのみ使用してください。
- 関係には上記のPコード（P131, P27等）のみ使用してください。自由記述は禁止です。
- 各関係には、根拠となる文書中のテキストをevidenceとして付与してください。
- headとtailにはentitiesのidを指定してください。
```

**Step 2: ユーザプロンプト構築**

以下を含むユーザプロンプトを構築する:
1. **Few-shot例**: 訓練データから選択した1文書のテキストと、その正解をJSON形式に変換した期待出力
2. **対象文書**: 抽出対象のテキスト

```
## 例
入力文書:
{few_shot文書のテキスト}

出力:
{few_shot文書の正解をJSON化したもの}

## 対象文書
{対象文書のテキスト}

上記の文書からエンティティと関係を抽出してください。
```

**Step 3: LLM呼び出し**

`EXTRACTION_SCHEMA` を `response_schema` として指定し、Gemini APIを1回呼び出す。レスポンスはJSON形式で `entities` と `relations` を含む。

**Step 4: 後処理（フィルタリング）**

1. **不正関係フィルタ**: `relations` 中の `relation` フィールドが35種類のPコードに含まれないものを除去
2. **不正エンティティタイプフィルタ**: `entities` 中の `type` が8種類のタイプに含まれないものを除去し、そのエンティティを参照する関係も除去

### 5.2 Mention-Span Constrained Entity Normalization（EntityNorm）

本手法は、Baselineにおけるエンティティアライメント失敗の問題を解決するために設計された。Baselineでは、LLMが生成するエンティティ名がGoldデータのmention名と一致しないことが多く（例: "東京都足立区" vs Gold "足立区"）、これがFPとFNの両方を増加させる要因となっている。

EntityNorm手法は、JacREDのvertexSet構造を模倣し、エンティティの**mentionレベルの抽出**と**正規化**を明示的に行うことで、この問題の軽減を図る。

#### Step 1: エンティティメンション抽出（1回のLLM呼び出し）

文書中の全てのエンティティメンション（言及）を、**正確なテキストスパン（verbatimな部分文字列）** とともに抽出する。

**システムプロンプト:**
```
あなたは日本語文書から固有表現（エンティティメンション）を抽出する専門家です。
文書中のすべてのエンティティメンションを正確に抽出してください。
```

**ユーザプロンプトの主要指示:**
```
## ルール
- mention_textは文書中のそのままの文字列（verbatimな部分文字列）でなければなりません。
  文書に含まれない文字列を記入しないでください。
- sentence_indexは、そのメンションが出現する文の番号（0始まり）を指定してください。
- 同一エンティティが複数回言及される場合（共参照）、同じcanonical_nameで
  グループ化してください。
  例: 「東京」「東京都」「同市」が同じ場所を指す場合、canonical_nameを「東京」に統一します。
- 文書中のすべてのエンティティメンションを漏れなく抽出してください。
```

出力は `SPAN_ENTITY_SCHEMA` に従い、各メンションの `mention_text`、`sentence_index`、`canonical_name`、`type` を含む。

**このステップの意義:**
- `mention_text` にverbatimなスパンを強制することで、文書中に実際に存在するテキストのみがエンティティ名として使用される
- `canonical_name` により、共参照関係にあるメンションを同一エンティティとして扱える
- `sentence_index` により、メンションの出現位置が記録される

#### Step 2: canonical_nameによるクラスタリング

抽出されたメンションを `canonical_name` の正規化結果（strip + lowercase）でグループ化し、エンティティクラスタを構築する。

```python
# entity_normalization.py の cluster_mentions() 関数
groups = {}
for mention in mentions:
    key = mention["canonical_name"].strip().lower()
    if key not in groups:
        groups[key] = {
            "canonical_name": mention["canonical_name"].strip(),
            "type": mention["type"],
            "mention_texts": [],
        }
    if mention["mention_text"] not in groups[key]["mention_texts"]:
        groups[key]["mention_texts"].append(mention["mention_text"])

# 各クラスタに "e0", "e1", ... のIDを付与
clusters = [{"id": f"e{i}", "canonical_name": g["canonical_name"],
             "type": g["type"], "mentions": g["mention_texts"]}
            for i, g in enumerate(groups.values())]
```

例えば、以下のようなメンションが抽出された場合:
- mention_text="アメリカ合衆国", canonical_name="アメリカ合衆国"
- mention_text="アメリカ", canonical_name="アメリカ合衆国"
- mention_text="米国", canonical_name="アメリカ合衆国"

これらは1つのクラスタ `{id: "e0", canonical_name: "アメリカ合衆国", mentions: ["アメリカ合衆国", "アメリカ", "米国"]}` にまとめられる。

**JacREDのvertexSetとの対応:**

JacREDのvertexSetでは、同一エンティティの異なるmention（例: "ジョー・ギブス" と "ギブス"）が1つのエンティティグループにまとめられている。EntityNormのクラスタリングはこの構造を模倣しており、エンティティアライメントの際に各クラスタのmention_text群とGoldのmention名群をマッチングできるようになる。

#### Step 3: クラスタ化エンティティを用いた関係抽出（1回のLLM呼び出し）

Step 2で得られたクラスタ化エンティティリストを含むプロンプトでLLMを呼び出し、エンティティ間の関係を抽出する。

**ユーザプロンプトの構造:**
```
## 例
入力文書:
{few_shot文書のテキスト}

出力:
{few_shot文書の正解をJSON化したもの}

## 対象文書
{対象文書のテキスト}

## 検出済みエンティティ（クラスタリング済み）
以下のエンティティが文書中に検出されました。各エンティティには複数のメンション
（言及表現）があります。

  - e0: アメリカ合衆国 (type=LOC, mentions=["アメリカ合衆国", "アメリカ", "米国"])
  - e1: ジョー・ギブス (type=PER, mentions=["ジョー・ギブス", "ギブス"])
  ...

上記のエンティティ間の関係を抽出してください。
- headとtailには上記のエンティティIDを使用してください。
- entitiesには上記のエンティティリストをそのまま使用してください。
- 関係の根拠となるテキストをevidenceとして付与してください。
```

**このステップの意義:**
- LLMにエンティティリストを事前に提示することで、エンティティの認識は完了済みとなり、関係抽出に集中できる
- 各エンティティの全mentionが表示されるため、LLMはエンティティに関するより豊富なコンテキスト情報を利用できる
- エンティティIDが固定されるため、LLMが勝手にIDを割り振る場合の不整合を回避できる

#### Step 4: 後処理（domain/range型制約）

Baselineと同一の後処理を適用する:

1. **不正関係フィルタ**: 35種類のPコードに含まれない関係を除去
2. **不正エンティティタイプフィルタ**: 8種類のタイプに含まれないエンティティの関係を除去
3. **Domain/Range型制約**: 訓練データで未観測の `(head_type, tail_type)` ペアを持つトリプルを除去

**Domain/Rangeとは何か:**

知識グラフやオントロジーの文脈において、**domain（定義域）** と **range（値域）** は関係（relation / property）に対する型制約を表す用語である:
- **Domain（定義域）**: その関係の **head（主語）** に許容されるエンティティタイプの集合。例えば、「生年月日」（P569）のdomainは `{PER}`（人物のみが生年月日を持つ）。
- **Range（値域）**: その関係の **tail（目的語）** に許容されるエンティティタイプの集合。例えば、「生年月日」（P569）のrangeは `{DAT}`（生年月日の値は日付である）。

本実験では、オントロジーで明示的に定義されたdomain/rangeではなく、**訓練データから経験的に観測されたtype pairの集合**を制約として使用する。

**制約テーブルの構築方法:**

訓練データ全体（1,400文書）をスキャンし、各関係Pコードについて、実際に出現した `(head_entity_type, tail_entity_type)` ペアの集合を収集する。

```python
# data_loader.py の build_constraint_table() 関数
constraint_table = defaultdict(set)
for doc in train_data:
    for label in doc["labels"]:
        h_type = doc["vertexSet"][label["h"]][0]["type"]
        t_type = doc["vertexSet"][label["t"]][0]["type"]
        constraint_table[label["r"]].add((h_type, t_type))
```

**フィルタの適用:**

```python
# extraction.py の apply_domain_range_constraints() 関数
for triple in candidates:
    allowed_pairs = constraint_table.get(triple.relation)
    if allowed_pairs is not None:
        if (triple.head_type, triple.tail_type) not in allowed_pairs:
            continue  # 除去（訓練データで観測されていない型ペア）
    # 保持
```

### 5.3 評価方法

#### エンティティアライメント（予測エンティティ → Gold vertexSet）

予測されたエンティティをGoldデータのvertexSetインデックスに対応付ける。3パスマッチングを以下の優先順で行う:

**Pass 1: 完全一致**
- 予測エンティティの `name` が、GoldのvertexSet中のいずれかのmentionの `name` と完全一致する場合にマッチ

**Pass 2: 正規化一致**
- 予測エンティティの `name` をUnicode NFKC正規化 + 小文字化 + 前後空白除去したものが、Goldのいずれかのmention nameの同様の正規化結果と一致する場合にマッチ

**Pass 3: 部分文字列一致**
- 正規化後の予測名がGold名の部分文字列であるか、またはGold名が予測名の部分文字列である場合にマッチ
- 複数候補がある場合は、重複する文字数（`min(len(pred), len(gold))`）が最大のものを優先
- 最小重複文字数は2文字（1文字のみの一致は無視）

**制約**: 各予測エンティティは最大1つのGoldエンティティにマッチする（1:1マッピング、先着順）。一度マッチしたGoldエンティティは以降のマッチ候補から除外される。

#### 関係評価

- **True Positive (TP)**: 予測トリプル `(head_id, relation, tail_id)` のhead, tailがともにGoldエンティティにアライメント済みで、かつGoldラベルに `(aligned_h_idx, aligned_t_idx, relation)` が存在する
- **False Positive (FP)**: 予測トリプルが以下のいずれかに該当:
  - headまたはtailがGoldエンティティにアライメントされていない（`entity_not_aligned`）
  - アライメントは成功したが、対応するGoldラベルが存在しない（`wrong_relation`）
- **False Negative (FN)**: Goldラベルのうち、いずれの予測トリプルにもマッチしなかったもの

#### 集計

10文書全体で **マイクロ平均（micro-average）** を計算する。

**マイクロ平均 vs マクロ平均:**

評価指標の集計方法には主に2種類ある:
- **マイクロ平均（micro-average）**: 全文書のTP, FP, FNを**合算してから**P/R/F1を計算する。文書ごとの重みは関係数に比例する（関係数の多い文書がスコアに与える影響が大きい）。本実験ではこちらを採用。
- **マクロ平均（macro-average）**: 各文書のP/R/F1を**個別に計算してから平均**する。文書ごとの重みは均等（関係数によらず各文書のスコアが等しく寄与する）。関係数が少ない文書のスコア変動が大きいため、サンプル数が少ない場合は不安定になりやすい。

本実験でマイクロ平均を採用した理由は、(a) DocRE分野の標準的な評価方式であること、(b) 10文書というサンプル数では文書あたりのスコア変動が大きく、マクロ平均は不安定になりやすいことである。

```
Precision = TP_total / (TP_total + FP_total)
Recall    = TP_total / (TP_total + FN_total)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

---

## 6. 結果

### 6.1 条件間比較表（集計）

モデル: `gemini-3-flash-preview`、`thinking_budget=2048`

| 条件 | P | R | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Baseline | 0.30 | 0.21 | 0.25 | 31 | 73 | 117 |
| EntityNorm | **0.38** | **0.22** | **0.28** | 32 | 52 | 116 |

EntityNormはBaselineに対して:
- **Precisionが+8ポイント改善**（0.30 → 0.38）
- **FPが29%削減**（73 → 52）
- **Recallは微増**（0.21 → 0.22）
- **F1が+3ポイント改善**（0.25 → 0.28）

### 6.2 Baseline: 文書別結果

| 文書 | P | R | F1 | TP | FP | FN | アライメント数 |
|---|---|---|---|---|---|---|---|
| ダニエル・ウールフォール | 1.00 | 0.58 | 0.74 | 7 | 0 | 5 | 8 |
| アンソニー世界を駆ける | 0.43 | 0.50 | 0.46 | 3 | 4 | 3 | 9 |
| 青ナイル州 | 0.43 | 0.15 | 0.22 | 3 | 4 | 17 | 11 |
| 小谷建仁 | 0.31 | 0.36 | 0.33 | 4 | 9 | 7 | 16 |
| 窪田僚 | 0.07 | 0.09 | 0.08 | 1 | 13 | 10 | 16 |
| イーオー | 0.11 | 0.10 | 0.11 | 1 | 8 | 9 | 17 |
| 堂山鉄橋 | 0.29 | 0.22 | 0.25 | 2 | 5 | 7 | 15 |
| 木村千歌 | 0.12 | 0.08 | 0.10 | 2 | 15 | 23 | 20 |
| バハン地区 | 0.30 | 0.11 | 0.16 | 3 | 7 | 25 | 14 |
| ジョー・ギブス | 0.38 | 0.31 | 0.34 | 5 | 8 | 11 | 29 |

### 6.3 EntityNorm: 文書別結果

| 文書 | P | R | F1 | TP | FP | FN | メンション数 | クラスタ数 | 関係数 | 制約後 |
|---|---|---|---|---|---|---|---|---|---|---|
| ダニエル・ウールフォール | 0.56 | 0.42 | 0.48 | 5 | 4 | 7 | 15 | 11 | 9 | 9 |
| アンソニー世界を駆ける | 0.40 | 0.33 | 0.36 | 2 | 3 | 4 | 16 | 15 | 7 | 5 |
| 青ナイル州 | 0.57 | 0.20 | 0.30 | 4 | 3 | 16 | 18 | 15 | 7 | 7 |
| 小谷建仁 | 0.80 | 0.36 | 0.50 | 4 | 1 | 7 | 23 | 19 | 12 | 5 |
| 窪田僚 | 0.13 | 0.09 | 0.11 | 1 | 7 | 10 | 21 | 19 | 12 | 8 |
| イーオー | 0.13 | 0.10 | 0.11 | 1 | 7 | 9 | 30 | 18 | 9 | 8 |
| 堂山鉄橋 | 0.17 | 0.11 | 0.13 | 1 | 5 | 8 | 26 | 19 | 9 | 6 |
| 木村千歌 | 0.15 | 0.08 | 0.11 | 2 | 11 | 23 | 25 | 21 | 18 | 13 |
| バハン地区 | 0.38 | 0.11 | 0.17 | 3 | 5 | 25 | 40 | 18 | 8 | 8 |
| ジョー・ギブス | 0.60 | 0.56 | 0.58 | 9 | 6 | 7 | 36 | 32 | 18 | 15 |

メンション数 = Step 1で抽出されたメンション総数、クラスタ数 = Step 2のクラスタリング後のエンティティ数、関係数 = Step 3で抽出された関係数、制約後 = domain/range制約適用後の最終関係数

### 6.4 エンティティアライメント統計

| 条件 | 予測エンティティ総数 | アライメント成功数 | アライメント率 |
|---|---|---|---|
| Baseline | 104（関係に使用） | 155 | --- |
| EntityNorm | 84（関係に使用） | 166 | --- |

注: `num_entities_aligned` はGold vertexSetへのアライメントに成功した予測エンティティの数であり、関係に使用された予測エンティティ数とは異なる場合がある。

### 6.5 アブレーションスタディ（モデル構成別比較）

5つのモデル構成について、同一の10文書（JacRED devセット）でBaseline / EntityNorm の両条件を実行した結果を以下に示す。

| Model Config | Baseline P | Baseline R | Baseline F1 | EntNorm P | EntNorm R | EntNorm F1 | Delta F1 |
|---|---|---|---|---|---|---|---|
| gemini-3-flash t=2048 | 0.298 | 0.209 | 0.246 | 0.381 | 0.216 | **0.276** | +0.030 |
| gemini-3-flash t=0 | 0.268 | 0.176 | 0.212 | 0.289 | 0.162 | 0.208 | -0.004 |
| gemini-2.5-flash t=2048 | 0.185 | 0.169 | 0.177 | 0.253 | 0.162 | **0.198** | +0.021 |
| gemini-2.5-flash t=0 | 0.190 | 0.155 | 0.171 | 0.277 | 0.155 | **0.199** | +0.028 |
| gemini-2.0-flash | 0.198 | 0.135 | 0.161 | 0.159 | 0.095 | 0.119 | -0.042 |

**主な知見:**

- **EntityNormは5構成中3構成でF1を改善**する。改善幅は+0.021から+0.030の範囲である
- **最良結果**: gemini-3-flash t=2048でF1=0.276（Delta F1=+0.030）を達成した。これは全構成中の最高F1である
- **Precisionの一貫した向上**: F1が改善した3構成では、いずれもPrecisionがBaselineを上回っている。エンティティ正規化により重複・変異エンティティが統合され、FPが削減されるためと考えられる
- **gemini-2.0-flashでの劣化**: Delta F1=-0.042と最大の劣化を示した。gemini-2.0-flashはthinking機能を持たず、EntityNormのStep 1（スパンエンティティ抽出）で正確なverbatimスパンとcanonical_nameの生成に十分なモデル能力が不足していると考えられる。EntityNorm手法はモデルの基本能力に依存し、能力が不十分な場合はパイプラインの複雑化がノイズを増幅させるリスクがある
- **thinking有無の影響**: gemini-3-flashではt=2048（thinking ON）がt=0（thinking OFF）を大きく上回る（F1: 0.276 vs 0.208）。一方、gemini-2.5-flashではt=0のEntNorm F1（0.199）がt=2048（0.198）とほぼ同等であり、thinkingの効果はモデル世代によって異なる

---

## 7. 分析

### 7.1 主な知見

1. **EntityNormはPrecisionを大幅改善**: FP数が73 → 52（29%削減）と顕著に改善された。特に小谷建仁文書ではFPが9 → 1と劇的に減少し、P=0.31 → 0.80と大幅向上した

2. **Recallの改善は限定的**: TP数は31 → 32（+1件のみ）で、Recallは0.21 → 0.22と微増にとどまった。FN数も117 → 116とほぼ横ばいである。これは、EntityNormがエンティティアライメント精度を向上させる一方で、関係の抽出網羅性自体は大きく改善しないことを示す

3. **ジョー・ギブス文書で最大のF1改善**: F1が0.34 → 0.58（+24ポイント）と、全文書中最大の改善を達成した。この文書はエンティティ数が31と最多であり、EntityNormのクラスタリングによる共参照解決の恩恵が大きかったと考えられる。TP数も5 → 9と大幅に増加した

4. **一部文書ではEntityNormが劣化**: ダニエル・ウールフォール文書ではBaselineのF1=0.74に対しEntityNormのF1=0.48と劣化した。BaselineがTP=7、FP=0という高精度を達成しているのに対し、EntityNormではFP=4が発生した。これはクラスタリング時のエンティティID再割り当てにより、関係抽出時にLLMが誤った関係を生成した可能性がある

5. **FPの主要パターン**:
   - **エンティティアライメント失敗** (`entity_not_aligned`): EntityNormでも依然として発生する。例: "ランカシャー州サッカー協会"がGoldに存在しない、"ソウル特別市"がGoldの"ソウル"とアライメントされない等
   - **関係方向の誤り / 関係タイプの誤り** (`wrong_relation`): 例: P6（首長）の方向が逆、P123（出版社）を誤って適用等

6. **FNの主要パターン**:
   - **行政区画の包含関係**: P131（行政区画）、P361（part of）、P527（has part）が大量にFNとなっている。特にバハン地区文書ではFN=25件のほとんどがこれらの関係であった
   - **制作者関係の未検出**: P170（制作者）がFNとなるケースが多い（木村千歌文書で顕著）
   - **逆方向関係ペア**: P361 (part of) と P527 (has part) の両方が正解に含まれるケースで、片方しか抽出できない

7. **domain/range制約の効果**: EntityNormのStep 3で抽出された関係110件のうち、制約適用後に84件（76%）が残った。約24%の不正な型ペアを持つ関係が除去されている

### 7.2 パイプライン段階別の候補数変化（EntityNorm、全10文書合計）

| 段階 | 候補数合計 |
|---|---|
| Step 1: メンション抽出 | 250 |
| Step 2: クラスタリング後エンティティ数 | 187 |
| Step 3: 関係抽出 | 110 |
| Step 4: domain/range制約後 | 84 |

メンション250件がクラスタリングにより187エンティティに統合され（平均1.34メンション/エンティティ）、そこから110件の関係が抽出され、制約適用後に84件が最終結果となっている。

### 7.3 条件別の長所・短所

| 観点 | Baseline | EntityNorm |
|---|---|---|
| Precision | 0.30 | **0.38** |
| Recall | 0.21 | 0.22 |
| F1 | 0.25 | **0.28** |
| API呼び出し回数/文書 | 1回 | 2回 |
| エンティティ共参照解決 | なし | あり（canonical_name） |
| エンティティアライメント品質 | 名前ベース | スパン + 名前ベース |
| 適用コスト | 低い | Baselineの約2倍 |

---

## 8. 再現方法

### 8.1 前提条件

- Python 3.10以上
- Google Gemini APIキー（[Google AI Studio](https://aistudio.google.com/)で取得）
- インターネット接続（API呼び出し用）

### 8.2 手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/auto-res2/kg-extraction-entity-normalization
cd kg-extraction-entity-normalization

# 2. JacREDデータセットを取得
git clone https://github.com/YoumiMa/JacRED /tmp/JacRED

# 3. 依存パッケージをインストール
pip install google-genai

# 4. APIキーを設定（2つの方法）
# 方法A: 環境変数（推奨）
export GEMINI_API_KEY="your-gemini-api-key"

# 方法B: ファイルから読み込み（デフォルトのパスを変更する場合はrun_experiment.pyのENV_PATHを編集）

# 5. 実験を実行
python3 run_experiment.py
```

**注意**: デフォルトでは `run_experiment.py` の `ENV_PATH` がDropbox上のファイルを参照するよう設定されている。環境変数 `GEMINI_API_KEY` を使用する場合は、`llm_client.py` の `load_api_key()` 呼び出し部分を `os.environ["GEMINI_API_KEY"]` に置き換えるか、`run_experiment.py` の `ENV_PATH` を適切なパスに変更する必要がある。

### 8.3 モデル・thinking設定の変更方法

`llm_client.py` を直接編集する:

```python
# llm_client.py の9行目
MODEL = "gemini-3-flash-preview"  # 変更先: "gemini-2.0-flash", "gemini-2.5-flash", etc.

# llm_client.py の41行目（call_gemini関数内）
thinking_config=ThinkingConfig(thinking_budget=2048),  # 0でOFF、2048でON
# gemini-2.0-flashを使う場合はこの行を削除する（thinking非対応のため）
```

### 8.4 JacREDデータのパス変更

デフォルトでは `/tmp/JacRED/` を参照する。変更する場合は `data_loader.py` の `load_jacred()` 関数の `base_path` 引数を変更する。

### 8.5 実行時間の目安

| 条件 | 概算時間（10文書） |
|---|---|
| Baseline | 約2分 |
| EntityNorm | 約4分（2回のLLM呼び出し/文書） |

注: API呼び出しの待機時間に大きく依存するため、上記は概算である。

---

## 9. ファイル構成

```
kg-extraction-entity-normalization/
  run_experiment.py        # メインスクリプト
  data_loader.py           # データ読み込み・選択
  llm_client.py            # Gemini API呼び出し
  prompts.py               # プロンプトテンプレート
  extraction.py            # 抽出ロジック
  evaluation.py            # 評価ロジック
  schemas.py               # JSON Schema定義
  entity_normalization.py  # エンティティ正規化ロジック
  results.json             # 最新の実験結果
  README.md                # 本ファイル
```

### 9.1 `run_experiment.py` --- メインスクリプト

**目的**: 実験全体のオーケストレーション（データ読み込み → 条件実行 → 結果比較・保存）。

**主要関数:**
- `run_condition(name, docs, few_shot, client, schema_info, extraction_fn="baseline", constraint_table=None)`:
  - 1つの実験条件（BaselineまたはEntityNorm）を全文書に対して実行し、文書別・集計のP/R/F1を算出する
  - `extraction_fn="baseline"` の場合は `run_baseline()` を呼び出し、`"entity_normalized"` の場合は `run_entity_normalized()` を呼び出す
  - 入力: 文書リスト、few-shot例、Geminiクライアント、スキーマ情報、抽出関数名、（任意）制約テーブル
  - 出力: `{"per_doc": [...], "aggregate": {...}}` の辞書
- `main()`:
  - データ読み込み（`load_jacred()`）、文書選択（`select_dev_docs()`）、few-shot選択（`select_few_shot()`）、制約テーブル構築（`build_constraint_table()`）を実行
  - Baseline, EntityNorm の2条件を順に実行し、結果を比較表示
  - `results.json` に全結果を保存

### 9.2 `data_loader.py` --- データ読み込み・選択

**目的**: JacREDデータセットの読み込み、実験用文書の選択、domain/range制約テーブルの構築。

**主要関数:**
- `load_jacred(base_path="/tmp/JacRED/") -> dict`:
  - train/dev/test の3分割JSONと、メタデータ（rel2id, ent2id, rel_info）を読み込む
  - 出力: `{"train": [...], "dev": [...], "test": [...], "rel2id": {...}, "ent2id": {...}, "rel_info": {...}}`
- `doc_to_text(doc) -> str`:
  - トークン化された文（`doc["sents"]`）を平文テキストに変換する。各文のトークンを結合し、さらに全文を結合する
- `char_count(doc) -> int`:
  - 文書の総文字数を計算する（全トークンの文字数合計）
- `select_dev_docs(dev_data, n=10) -> list`:
  - devセットから文字数順にソートし、量子位置で `n` 文書を選択する層化サンプリング
- `select_few_shot(train_data) -> dict`:
  - 訓練データからfew-shot例に適した文書を選択する（150-250文字、5-12エンティティ、3-15ラベル）
- `format_few_shot_output(doc) -> dict`:
  - JacRED文書のvertexSetとlabelsから、EXTRACTION_SCHEMAに準拠したJSON形式の出力例を生成する
- `build_constraint_table(train_data) -> dict`:
  - 訓練データ全体から、各関係Pコードに対する観測済み `(head_type, tail_type)` ペアの集合を構築する

### 9.3 `llm_client.py` --- Gemini API呼び出し

**目的**: Google Gemini APIの呼び出し、Structured Outputs対応、リトライロジック。

**主要関数・定数:**
- `MODEL = "gemini-3-flash-preview"`: 使用するモデルID（変更時はここを編集）
- `load_api_key(env_path) -> str`: `.env` ファイルから `GEMINI_API_KEY` を読み込む
- `create_client(api_key) -> genai.Client`: Geminiクライアントを生成する
- `call_gemini(client, system_prompt, user_prompt, response_schema, temperature=0.2, max_retries=3) -> dict`:
  - Gemini APIを呼び出し、Structured OutputsでJSON応答を取得してパース済み辞書として返す
  - `GenerateContentConfig` に `response_mime_type="application/json"` と `response_schema` を設定
  - `ThinkingConfig(thinking_budget=2048)` がハードコードされている（変更時はここを編集）
  - 失敗時は指数バックオフ（2^(attempt+1) 秒）でリトライ

### 9.4 `prompts.py` --- プロンプトテンプレート

**目的**: 全LLM呼び出し用のプロンプト構築ロジック。

**主要定数:**
- `RELATION_JAPANESE`: 35種類の関係PコードからJapanese descriptionへのマッピング辞書
- `ENTITY_TYPES_JAPANESE`: 8種類のエンティティタイプからJapanese descriptionへのマッピング辞書

**主要関数:**
- `build_system_prompt(rel_info) -> str`: エンティティタイプ・関係タイプを含むシステムプロンプトを構築する。`rel_info` は `{Pコード: 英語名}` の辞書（JacREDメタデータ由来）
- `build_extraction_prompt(doc_text, few_shot_text, few_shot_output, mode="baseline") -> str`: 抽出用ユーザプロンプトを構築する。`mode="recall"` の場合はRecall重視の指示を追加する
- `build_span_entity_prompt(doc_text, few_shot_text) -> str`: EntityNorm Step 1のスパンエンティティ抽出プロンプトを構築する。mention_textがverbatimなスパンであること、canonical_nameによる共参照グループ化ルール等を含む
- `build_verification_prompt(doc_text, candidates, entity_map, rel_info) -> str`: Stage 2検証用プロンプトを構築する（本実験ではEntityNorm条件では未使用だが、ベースリポジトリのTwo-Stage条件用に存在する）

### 9.5 `extraction.py` --- 抽出ロジック

**目的**: Baseline・EntityNorm条件の抽出パイプライン全体を実装する。

**主要クラス:**
- `Triple`: データクラス。抽出されたトリプルを表現する
  - フィールド: `head`（エンティティID）, `head_name`, `head_type`, `relation`（Pコード）, `tail`, `tail_name`, `tail_type`, `evidence`

**主要関数:**
- `run_baseline(doc, few_shot, client, schema_info) -> (entities, triples)`:
  - Baseline条件を1文書に対して実行する。システムプロンプト構築 → ユーザプロンプト構築（mode="baseline"） → LLM呼び出し → パース → フィルタ
- `run_entity_normalized(doc, few_shot, client, schema_info, constraint_table) -> (entities, triples, stats)`:
  - EntityNorm条件を1文書に対して実行する。Step 1（スパンエンティティ抽出） → Step 2（クラスタリング） → Step 3（クラスタ化エンティティを用いた関係抽出） → Step 4（後処理・制約適用）
  - `stats` にはパイプライン各段階の数値を記録: `{"num_mentions": N, "num_clusters": M, "num_triples": K, "after_constraints": L}`
- `run_proposed(doc, few_shot, client, schema_info, constraint_table) -> (entities, triples, stats)`:
  - ベースリポジトリから継承したTwo-Stage条件（本実験では使用しないが参考用に存在）
- `filter_invalid_labels(triples, valid_relations) -> list[Triple]`:
  - 不正なPコードを持つトリプルを除去する
- `filter_invalid_entity_types(triples, valid_types) -> list[Triple]`:
  - 不正なエンティティタイプを持つトリプルを除去する
- `apply_domain_range_constraints(triples, constraint_table) -> list[Triple]`:
  - 訓練データで未観測の `(head_type, tail_type)` ペアを持つトリプルを除去する
- `_verify_candidates(doc, candidates, entity_id_to_name, client, schema_info, batch_size=10) -> list[Triple]`:
  - Two-Stage条件用のバッチ検証（本実験のEntityNorm条件では未使用）

### 9.6 `evaluation.py` --- 評価ロジック

**目的**: エンティティアライメントとP/R/F1の算出。

**主要関数:**
- `align_entities(predicted_entities, gold_vertex_set) -> dict[str, int]`:
  - 予測エンティティをGold vertexSetにアライメントする（3パスマッチング: 完全一致 → 正規化一致 → 部分文字列一致）
  - 出力: `{予測エンティティID: Gold vertexSetインデックス}`
- `evaluate_relations(predicted_triples, gold_labels, entity_alignment) -> dict`:
  - アライメント結果を用いて予測トリプルをGoldラベルと照合し、TP/FP/FN/P/R/F1 を算出する
  - FP詳細（理由: `entity_not_aligned` or `wrong_relation`）とFN詳細を含む
- `aggregate_results(per_doc) -> dict`:
  - 文書別結果リストからマイクロ平均のP/R/F1を算出する

### 9.7 `schemas.py` --- JSON Schema定義

**目的**: Gemini Structured Outputs用のJSON Schema定義。

**定数:**
- `EXTRACTION_SCHEMA`: 抽出用スキーマ。`entities`（id, name, typeの配列）と `relations`（head, relation, tail, evidenceの配列）を要求する。Baseline・EntityNorm Step 3の両方で使用
- `SPAN_ENTITY_SCHEMA`: スパンエンティティ抽出用スキーマ。`entity_mentions`（mention_text, sentence_index, canonical_name, typeの配列）を要求する。EntityNorm Step 1で使用
- `VERIFICATION_SCHEMA`: 検証用スキーマ。`decisions`（candidate_index, keepの配列）を要求する。ベースリポジトリのTwo-Stage条件用（本実験では未使用）

### 9.8 `entity_normalization.py` --- エンティティ正規化ロジック

**目的**: EntityNorm手法のコアロジック。メンションのクラスタリングとクラスタ化エンティティを用いた関係抽出プロンプトの構築。

**主要関数:**
- `cluster_mentions(mentions: list[dict]) -> list[dict]`:
  - メンションリストを `canonical_name` の正規化結果（strip + lowercase）でグループ化する
  - 出力: クラスタリスト。各クラスタは `{id: "eN", canonical_name: str, type: str, mentions: list[str]}` の辞書
  - 同一canonical_nameの重複mention_textは排除される
  - エンティティタイプは最初に出現したメンションのタイプが採用される
- `build_clustered_entity_prompt(doc_text, clusters, few_shot_text, few_shot_output) -> str`:
  - クラスタ化エンティティリストを含む関係抽出用プロンプトを構築する
  - 各クラスタのID、canonical_name、タイプ、全mention一覧をプロンプトに埋め込む
  - LLMに対して上記エンティティID間の関係を抽出するよう指示する

### 9.9 `results.json` --- 最新の実験結果

**目的**: 最後に実行された実験の全結果をJSON形式で保存する。

**構造:**
```json
{
  "experiment": {
    "model": "gemini-3-flash-preview",
    "num_docs": 10,
    "few_shot_doc": "スタッド (バンド)",
    "timestamp": "2026-02-04T13:40:07.626632"
  },
  "conditions": {
    "baseline": {
      "per_doc": [{"title": "...", "precision": 0.6, ...}, ...],
      "aggregate": {"precision": 0.30, "recall": 0.21, "f1": 0.25, ...}
    },
    "entity_normalization": {
      "per_doc": [{"title": "...", "precision": 0.56, ..., "stats": {...}}, ...],
      "aggregate": {"precision": 0.38, "recall": 0.22, "f1": 0.28, ...}
    }
  }
}
```

---

## 10. 参考文献

1. Ma, Y., Tanaka, J., & Araki, M. **"Building a Japanese Document-Level Relation Extraction Dataset Assisted by Cross-Lingual Transfer."** *Proceedings of LREC-COLING 2024.*
   - JacREDデータセットの構築論文。本実験のデータセット。

2. Yao, Y., Ye, D., Li, P., Han, X., Lin, Y., Liu, Z., Liu, Z., Huang, L., Zhou, J., & Sun, M. **"DocRED: A Large-Scale Document-Level Relation Extraction Dataset."** *Proceedings of ACL 2019.*
   - JacREDの基となった英語DocREデータセット。

3. Tan, C., Zhao, W., Wei, Z., & Huang, X. **"Document-level Relation Extraction: A Survey."** *arXiv preprint, 2023.*
   - 文書レベル関係抽出のサーベイ論文。

4. Li, D., Liu, Y., & Sun, M. **"A Survey on LLM-based Generative Information Extraction."** *arXiv preprint, 2024.*
   - LLMによる情報抽出のサーベイ論文。

5. Giorgi, J., Bader, G., & Wang, B. **"End-to-end Named Entity Recognition and Relation Extraction using Pre-trained Language Models."** *arXiv preprint, 2019.*
   - 事前学習言語モデルを用いたend-to-end NER+RE。

6. Dagdelen, J., Dunn, A., Lee, S., Walker, N., Rosen, A., Ceder, G., Persson, K., & Jain, A. **"Structured information extraction from scientific text with large language models."** *Nature Communications, 2024.*
   - LLMによる科学文献からの構造化情報抽出。

7. Willard, B., & Louf, R. **"Generating Structured Outputs from Language Models."** *arXiv preprint, 2025.*
   - 言語モデルからの構造化出力生成手法。

8. Harnoune, A., Rhanoui, M., Asri, B., Zellou, A., & Yousfi, S. **"Information extraction pipelines for knowledge graphs."** *Knowledge and Information Systems (Springer), 2022.*
   - 知識グラフ構築のための情報抽出パイプライン。

9. Mintz, M., Bills, S., Snow, R., & Jurafsky, D. **"Distant supervision for relation extraction without labeled data."** *Proceedings of ACL 2009.*
   - ラベルなしデータからの遠隔教師あり関係抽出。

10. Lu, Y., Liu, Z., & Huang, L. **"Cross-Lingual Structure Transfer for Relation and Event Extraction."** *Proceedings of ACL 2019.*
    - 言語間構造転移による関係・イベント抽出。
