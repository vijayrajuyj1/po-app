from typing import List
from apps.ai.schemas import AIGenerationRequestSchema, AIGenerationResponseSchema


class AIService:
    @staticmethod
    async def generate_extraction_prompt(payload: AIGenerationRequestSchema) -> AIGenerationResponseSchema:
        """
        Generate extraction instructions based on question and keywords.
        This is a deterministic template-based generator intended as a placeholder for a real LLM.
        Replace with provider-specific calls if needed.
        """
        keywords_section = ""
        if payload.keywords:
            keywords_str = ", ".join(sorted(set(k.strip() for k in payload.keywords if k.strip())))
            keywords_section = f"\nKeywords to prioritize: {keywords_str}."
        instructions = (
            f"Extract the answer to the following question from the document in a precise and deterministic way:\n"
            f"Question: {payload.question.strip()}.{keywords_section}\n\n"
            f"Instructions:\n"
            f"1) Search the document for the question context and relevant terminology.\n"
            f"2) Prefer values near section headers or tables closely matching the question.\n"
            f"3) If multiple values exist, choose the most recent or the one in the primary contract/invoice body.\n"
            f"4) Return only the extracted value as plain text. Do not include units unless explicitly present in source.\n"
            f"5) If the value is not found, return an empty string."
        )
        return AIGenerationResponseSchema(extraction_instructions=instructions)


